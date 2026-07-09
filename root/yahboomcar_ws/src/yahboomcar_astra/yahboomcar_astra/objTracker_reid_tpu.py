#ros lib
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from yahboomcar_msgs.msg import Position
from cv_bridge import CvBridge
#common lib
import cv2
from yahboomcar_astra.astra_common import *
import tflite_runtime.interpreter as tflite
print("import done")


class EdgeTPUDetector:
    '''Coral Edge TPU 目标检测器（SSD MobileNet v2 COCO postprocess），同 objTracker_tpu.py。
    detect_candidates() 与其 detect() 的区别：返回该类别所有过阈值的框（按分数降序，
    截断到 max_candidates），供上层做外观匹配挑人，而不是只留最高分那个。'''

    def __init__(self, model_path, label_path):
        self.labels = self._load_labels(label_path)
        delegate = tflite.load_delegate('libedgetpu.so.1')
        self.interpreter = tflite.Interpreter(
            model_path=model_path, experimental_delegates=[delegate])
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        _, self.in_h, self.in_w, _ = self.input_details[0]['shape']
        print("edgetpu det model loaded:", model_path)

    @staticmethod
    def _load_labels(label_path):
        labels = {}
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    labels[parts[1]] = int(parts[0])
                else:
                    labels[parts[0]] = len(labels)
        return labels

    def label_id(self, target_label):
        return self.labels.get(target_label, -1)

    def detect_candidates(self, frame, conf_thr, target_id, max_candidates):
        '''返回 [(box(x1,y1,x2,y2), score), ...]，target_id 类且 score>=conf_thr，按分数降序截断'''
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.in_w, self.in_h))
        input_data = np.expand_dims(resized, axis=0).astype(np.uint8)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
        count = int(self.interpreter.get_tensor(self.output_details[3]['index'])[0])
        h, w = frame.shape[:2]
        candidates = []
        for i in range(count):
            if int(classes[i]) == target_id and scores[i] >= conf_thr:
                ymin, xmin, ymax, xmax = boxes[i]
                box = (int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h))
                candidates.append((box, float(scores[i])))
        candidates.sort(key=lambda c: c[1], reverse=True)
        return candidates[:max_candidates]


class EdgeTPUEmbedder:
    '''行人 Re-ID 嵌入器。默认用 Tencent Youtu Lab 的 person_reid_youtu_2021nov（ResNet50
    backbone，来自 opencv_zoo/PINTO_model_zoo，Apache 2.0），本项目自己重新做的 INT8
    量化+edgetpu 编译（PINTO 现成的 INT8 tflite 量化校准有 bug，输入按"原始像素直接透传"
    校准，而这个模型实际要求先做 ImageNet 归一化，导致同人/不同人相似度完全分不开；
    自己用正确的归一化+校准集重新量化后，同人 0.46~0.86 / 不同人 -0.02~0.32，真机
    Edge TPU 实测区分度良好）。输入 [1,256,128,3] uint8 RGB（256高×128宽，人体竖直裁剪
    比例），输出 768 维特征。代价：模型比旧的大很多（26MB vs 3.3MB），8MB 片上 SRAM
    缓存装不下，单次 embed() 实测 ~51ms（旧模型 ~3.9ms）——曾试过隔几帧才嵌入一次+纯位置
    连续性顶帧的省算力方案，但连续性判断本身不做外观校验，实测两人交叉/靠近时会在未嵌入
    的那几帧悄悄跳去别人身上（且 debug_sim 日志看不到，因为根本没调用嵌入器）；改回每帧
    都真跑嵌入器校验，用发布频率/max_candidates 兜住算力而不是跳过校验。

    旧模型 mobilenet_v1_1.0_224_quant_embedding_extractor（100%可缓存、~4ms/次，但
    区分度弱、对换角度/光照敏感，见 memory）留作备选，传 --ros-args -p
    det_model_path:=.../models/reid/det_reid_edgetpu.tflite -p
    emb_model_path:=.../models/reid/emb_reid_edgetpu.tflite 可切回（注意 det/emb 必须
    成对使用同一次 co-compile 产物，reid_youtu/ 和 reid/ 两个目录不能混用）。'''

    def __init__(self, model_path):
        delegate = tflite.load_delegate('libedgetpu.so.1')
        self.interpreter = tflite.Interpreter(
            model_path=model_path, experimental_delegates=[delegate])
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        _, self.in_h, self.in_w, _ = self.input_details[0]['shape']
        print("edgetpu emb model loaded:", model_path)

    def embed(self, frame, box):
        x1, y1, x2, y2 = box
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.in_w, self.in_h))
        input_data = np.expand_dims(resized, axis=0).astype(np.uint8)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        scale, zero = self.output_details[0]['quantization']
        raw = self.interpreter.get_tensor(self.output_details[0]['index'])[0].reshape(-1).astype(np.float32)
        vec = (raw - zero) * scale
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm
        return vec


def cosine_sim(a, b):
    return float(np.dot(a, b))


def center_dist(box_a, box_b):
    ax = (box_a[0] + box_a[2]) / 2.0
    ay = (box_a[1] + box_a[3]) / 2.0
    bx = (box_b[0] + box_b[2]) / 2.0
    by = (box_b[1] + box_b[3]) / 2.0
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


class Object_Identify_ReID(Node):
    '''检测器 + Re-ID 外观锁定：只跟一个被锁定的人，发布话题/字段与 objTracker_tpu.py
    完全一致（/Current_point 的 Position、/image_raw），身份状态全在节点内部，
    下游 person_goal_bridge.py / objControl.py 零改动。'''

    def __init__(self, name):
        super().__init__(name)
        self.pub_position = self.create_publisher(Position, "/Current_point", 10)
        self.pub_img = self.create_publisher(Image, '/image_raw', 500)
        self.sub_lock = self.create_subscription(Bool, "/Reid_Lock", self.on_lock_cmd, 1)
        self.bridge = CvBridge()
        self.end = 0
        self.hit_count = 0
        self.template = None      # 锁定身份的 EMA 嵌入向量，None=未锁定
        self.last_box = None      # 上一次锁定框，供 tie-break/连续性用
        self.force_relock = False
        self.pending_lock_box = None    # 自动初始锁定：正在计时观察的候选框（够近+没换人）
        self.pending_lock_since = 0.0   # 上面那个候选框从什么时候开始持续满足条件
        self.declare_param()
        self.detector = EdgeTPUDetector(self.det_model_path, self.label_path)
        self.embedder = EdgeTPUEmbedder(self.emb_model_path)
        self.target_id = self.detector.label_id(self.target_label)
        if self.target_id < 0:
            print("WARN: target_label '%s' not in labels, no detection will match" % self.target_label)
        self.capture = cv.VideoCapture(0)
        self.timer = self.create_timer(0.001, self.on_timer)
        print("init done")

    def declare_param(self):
        self.declare_parameter("target_label", "person")
        self.target_label = self.get_parameter('target_label').get_parameter_value().string_value
        self.declare_parameter("conf_threshold", 0.5)
        self.conf_threshold = self.get_parameter('conf_threshold').get_parameter_value().double_value
        self.declare_parameter("det_model_path",
            "/root/yahboomcar_ws/src/yahboomcar_astra/yahboomcar_astra/models/reid_youtu/det_reid_edgetpu.tflite")
        self.det_model_path = self.get_parameter('det_model_path').get_parameter_value().string_value
        self.declare_parameter("emb_model_path",
            "/root/yahboomcar_ws/src/yahboomcar_astra/yahboomcar_astra/models/reid_youtu/emb_reid_edgetpu.tflite")
        self.emb_model_path = self.get_parameter('emb_model_path').get_parameter_value().string_value
        self.declare_parameter("label_path",
            "/root/yahboomcar_ws/src/yahboomcar_astra/yahboomcar_astra/models/coco_labels.txt")
        self.label_path = self.get_parameter('label_path').get_parameter_value().string_value
        self.declare_parameter("show_image", True)
        self.show_image = self.get_parameter('show_image').get_parameter_value().bool_value
        self.declare_parameter("min_hits", 3)   # 连续 N 帧命中才发目标，滤掉一闪而过的误检
        self.min_hits = self.get_parameter('min_hits').get_parameter_value().integer_value
        self.declare_parameter("sim_floor", 0.6)   # 外观相似度硬门槛；0.4 实测太松——真目标那一帧漏检时，
        # 剩下随便一个候选只要过 0.4 就被当"确认"接受，之后还会 EMA 写回模板自我强化坐实误锁，故提高
        self.sim_floor = self.get_parameter('sim_floor').get_parameter_value().double_value
        self.declare_parameter("max_candidates", 3)   # 每次跑嵌入器最多处理的候选数上限，控制单帧TPU负载
        self.max_candidates = self.get_parameter('max_candidates').get_parameter_value().integer_value
        self.declare_parameter("ema_alpha", 0.1)   # 模板更新步长，只在高置信匹配时小步更新
        self.ema_alpha = self.get_parameter('ema_alpha').get_parameter_value().double_value
        self.declare_parameter("tie_margin", 0.05)   # 相似度差在此范围内视为并列，按空间连续性 tie-break
        self.tie_margin = self.get_parameter('tie_margin').get_parameter_value().double_value
        self.declare_parameter("ema_min_sim", 0.65)   # 只有匹配相似度明显高于 sim_floor 才更新模板，避免边界误匹配把模板带偏
        self.ema_min_sim = self.get_parameter('ema_min_sim').get_parameter_value().double_value
        self.declare_parameter("debug_sim", False)   # 打印每个候选的相似度，供车上调参诊断
        self.debug_sim = self.get_parameter('debug_sim').get_parameter_value().bool_value
        self.declare_parameter("lock_min_size", 300)   # 自动初始锁定：候选框最长边(像素)要不小于这个值才算"离得够近"
        self.lock_min_size = self.get_parameter('lock_min_size').get_parameter_value().integer_value
        self.declare_parameter("lock_stable_time", 3.0)   # 上面那个候选框要持续满足条件这么多秒才真正锁定
        self.lock_stable_time = self.get_parameter('lock_stable_time').get_parameter_value().double_value
        self.declare_parameter("lock_stable_dist", 60.0)   # 候选框中心帧间移动不超过这个像素距离，才算"同一个人还站在原地"
        self.lock_stable_dist = self.get_parameter('lock_stable_dist').get_parameter_value().double_value

    def on_lock_cmd(self, msg):
        '''收到 /Reid_Lock=True：下一帧强制重新锁定当前最突出的候选（覆盖旧模板），用于换人/纠错'''
        if msg.data:
            self.force_relock = True

    @staticmethod
    def _pick_initial(candidates):
        '''手动锁定(/Reid_Lock)专用：选面积最大的候选（离得最近/最突出的人），立即锁定不等待。
        自动初始锁定另有站稳计时逻辑，见 on_timer，不走这个函数。'''
        def area(c):
            x1, y1, x2, y2 = c[0]
            return (x2 - x1) * (y2 - y1)
        return max(candidates, key=area)

    def _match(self, frame, candidates):
        '''对所有候选跑嵌入器，取相似度过 sim_floor 里最高的一个；
        相似度接近（tie_margin 内）时按离上一帧锁定框更近的 tie-break，减少帧间抖动。'''
        best = None
        debug_sims = []
        for box, score in candidates:
            vec = self.embedder.embed(frame, box)
            if vec is None:
                continue
            sim = cosine_sim(vec, self.template)
            if self.debug_sim:
                debug_sims.append((box, sim))
            if sim < self.sim_floor:
                continue
            if best is None or sim > best[2] + self.tie_margin:
                best = (box, score, sim, vec)
            elif abs(sim - best[2]) <= self.tie_margin:
                if center_dist(box, self.last_box) < center_dist(best[0], self.last_box):
                    best = (box, score, sim, vec)
        if self.debug_sim:
            picked = best[0] if best is not None else None
            print("sims:", ["%s:%.3f%s" % (b, s, "*" if b == picked else "") for b, s in debug_sims])
        return best

    def on_timer(self):
        ret, frame = self.capture.read()
        if not ret:
            return
        frame = cv.resize(frame, (640, 480))
        start = time.time()
        fps = 1 / (start - self.end)
        self.end = start

        candidates = self.detector.detect_candidates(
            frame, self.conf_threshold, self.target_id, self.max_candidates)

        box = None
        if self.force_relock and candidates:
            # 手动触发(/Reid_Lock)：立即锁定当前最突出的人，不用等站稳——这就是要的"马上换人"效果
            lock_box, _ = self._pick_initial(candidates)
            self.template = self.embedder.embed(frame, lock_box)
            self.last_box = lock_box
            self.force_relock = False
            self.pending_lock_box = None
            box = lock_box
        elif self.template is None:
            # 自动初始锁定：不是一上来就锁最大候选，而是要"离得够近(框够大)+站稳 lock_stable_time 秒"
            # 才当真——相当于把"走近站定"当成免模型的手势触发，防止刚开机随便一个路人就被锁定
            big = [c for c in candidates
                   if max(c[0][2] - c[0][0], c[0][3] - c[0][1]) >= self.lock_min_size]
            biggest = max(big, key=lambda c: (c[0][2] - c[0][0]) * (c[0][3] - c[0][1])) if big else None
            now = time.time()
            if biggest is None:
                self.pending_lock_box = None
            elif (self.pending_lock_box is not None
                  and center_dist(biggest[0], self.pending_lock_box) <= self.lock_stable_dist):
                self.pending_lock_box = biggest[0]   # 同一个人还站在原地，位置微调但计时不清零
                if now - self.pending_lock_since >= self.lock_stable_time:
                    self.template = self.embedder.embed(frame, biggest[0])
                    self.last_box = biggest[0]
                    box = biggest[0]
                    self.pending_lock_box = None
            else:
                self.pending_lock_box = biggest[0]   # 新出现的人/换了个人站过来，重新计时
                self.pending_lock_since = now
        elif self.template is not None and candidates:
            best = self._match(frame, candidates)
            if best is not None:
                box, _, sim, vec = best
                self.last_box = box
                # 只在相似度明显高于 sim_floor(ema_min_sim) 时才更新模板，
                # 边界匹配(刚过 sim_floor)可以用来发布位置，但不写回模板——防止一次误匹配把模板带偏
                if sim >= self.ema_min_sim:
                    self.template = (1 - self.ema_alpha) * self.template + self.ema_alpha * vec
                    norm = np.linalg.norm(self.template)
                    if norm > 1e-6:
                        self.template = self.template / norm

        if box is not None:
            self.hit_count += 1
        else:
            self.hit_count = 0

        if box is not None and self.hit_count >= self.min_hits:
            (x1, y1, x2, y2) = box
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            size = max(x2 - x1, y2 - y1)
            position = Position()
            position.anglex = cx * 1.0
            position.angley = cy * 1.0
            position.distance = size * 1.0
            self.pub_position.publish(position)
        # 每帧都发 /image_raw：colorTracker 的 execute() 只在 /image_raw 回调里跑
        self.pub_img.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))
        if self.show_image:
            cv.putText(frame, "FPS : " + str(int(fps)), (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 1)
            for cbox, _ in candidates:
                is_locked = (box is not None and cbox == box)
                is_pending = (self.template is None and self.pending_lock_box is not None
                              and cbox == self.pending_lock_box)
                if is_locked:
                    color = (0, 255, 0) if self.hit_count >= self.min_hits else (0, 255, 255)
                elif is_pending:
                    color = (255, 255, 0)   # 正在倒计时观察的候选（还没正式锁定）
                    remain = max(0.0, self.lock_stable_time - (time.time() - self.pending_lock_since))
                    cv.putText(frame, "locking in %.1fs" % remain, (cbox[0], max(cbox[1] - 8, 0)),
                               cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                else:
                    color = (128, 128, 128)   # 候选但非锁定目标（如误检的椅子腿、旁人）
                cv2.rectangle(frame, (cbox[0], cbox[1]), (cbox[2], cbox[3]), color, 2)
            cv.imshow('frame', frame)
            if (cv.waitKey(10) & 0xFF) in (ord('q'), 113):
                self.capture.release()
                cv.destroyAllWindows()


def main():
    rclpy.init()
    object_identify = Object_Identify_ReID("ObjectIdentifyReID")
    print("start it")
    rclpy.spin(object_identify)
