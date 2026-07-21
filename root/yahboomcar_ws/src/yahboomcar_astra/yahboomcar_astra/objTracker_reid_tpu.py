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

# 新 Market-1501 Re-ID 嵌入器(osnet/mobilenetv2)要求 Python 侧先做 ImageNet 归一化再量化
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _load_edgetpu_delegate():
    '''Coral USB 冷启动要做固件握手：第一次 load_delegate 必然抛空 ValueError，设备从
    1a6e:089a 重枚举成 18d1:9302(已认领)后重试一次即成功（见 memory tpu-reid-plan；车重启/
    冷启动必复现，之前 Coral 已被预热才没暴露）。不重试会直接崩在 delegate 加载。'''
    try:
        return tflite.load_delegate('libedgetpu.so.1')
    except ValueError:
        return tflite.load_delegate('libedgetpu.so.1')


class EdgeTPUDetector:
    '''Coral Edge TPU 目标检测器（SSD MobileNet v2 COCO postprocess），同 objTracker_tpu.py。
    detect_candidates() 与其 detect() 的区别：返回该类别所有过阈值的框（按分数降序，
    截断到 max_candidates），供上层做外观匹配挑人，而不是只留最高分那个。'''

    def __init__(self, model_path, label_path):
        self.labels = self._load_labels(label_path)
        delegate = _load_edgetpu_delegate()
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
    '''行人 Re-ID 嵌入器。默认用 **reid_youtu**（未剪枝 Youtu lite，ResNet50 主干，多源训练
    Market+Duke+MSMT17+CUHK03，来自 opencv_zoo `person_reid_youtu_2021nov`，本项目自己重新
    量化+与 SSD co-compile，产物 models/reid_youtu/）。输入 [1,256,128,3] **uint8** RGB(256高×
    128宽,归一化烘焙进图内)，输出 768 维。真机单次 embed() ~60ms(~4.4Hz)——精度最稳(离线贴地
    逐帧顶替率 4%,是下限)，代价是慢；后续攻云台跟踪期间用它打底，帧率之后再优化。

    **预处理**：embed() 按输入 dtype 分支——uint8(本 Youtu/MobileNetV1)走原始像素直喂；int8
    (剪枝档/osnet)走 Python 侧 ImageNet 归一化 (pixel/255-mean)/std 再量化。裁剪统一用"以人框
    中心、固定 1:2 竖直躯干条"(见 embed())，对框变宽/贴近切边免疫、贴合 Market 裁片约定。

    **贴地视角是 OOD**：小车摄像头贴地仰视是 Market 训练分布外——用车上实拍帧离线评测(见
    benchmarks/bench_prune_sweep.py + docs devlog §8.9)：osnet 系(单源 Market)旁人与目标重叠、
    **贴地不可用**；只有多源 Youtu ResNet50 系能分。**贴地泛化≈容量、剪不动**——想要 base 级只能用 base。

    每帧对所有候选都真跑嵌入器(不隔帧)——隔帧方案有"非嵌入帧无外观校验"安全漏洞，已弃。

    备选（改 det_model_path/emb_model_path 参数切换，每对独立 co-compile 不能混用）：
      models/reid_youtu_p70/ Youtu剪70% ~15.6ms(~18Hz) 顶替率13%（帧率优先，已进镜像）
      models/reid_youtu_p50/ Youtu剪50% ~26ms 顶替率7%（需自己 co-compile 落盘）
      models/reid_osnet05|osnet075|mnv2/ 单源 Market——**贴地实测不可用，仅平视场景备选**
      models/reid/           MobileNetV1 ~3.7ms（通用 ImageNet 特征，最弱，uint8 输入）'''

    def __init__(self, model_path):
        delegate = _load_edgetpu_delegate()
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
        if x2 <= x1 or y2 <= y1:
            return None
        # 喂给嵌入器的裁片：**只收窄，绝不撑宽**——宽度取 min(人框宽, 高*1/2)，以框中心对齐。
        # 为什么收窄：直接把人框 resize 成 128x256 会随框形状畸变。张开手、或人走太近被画面
        # 切边，人框被撑成宽矩形(W/H 1.3+)，横向压扁近 3 倍——车上实测同一人相似度从 0.99
        # 掉到 0.80，比旁人(0.72)还低，任何 sim_floor 都分不开。收成竖条后这类框回到 0.96+。
        # 为什么不撑宽：曾经强制取 1:2(窄框也撑到 高/2)，结果车放到地上、人站远时人框又窄又高
        # (W/H 0.27)，撑宽等于往裁片里灌了 40%+ 背景，相似度反而塌到 0.5——而紧贴人体的窄框
        # 本来就已经是 Market-1501 的裁片约定，原样用即可。
        # 注意：只改喂给嵌入器的几何，发布位置/tie-break 仍用原始人框。
        fw = frame.shape[1]
        col_w = min(float(x2 - x1), (y2 - y1) * self.in_w / self.in_h)
        cx = (x1 + x2) / 2.0
        cx1, cx2 = cx - col_w / 2.0, cx + col_w / 2.0
        if cx1 < 0:                       # 竖条超出画面就整体滑回画面内，尽量保住比例
            cx1, cx2 = 0.0, min(col_w, float(fw))
        elif cx2 > fw:
            cx1, cx2 = max(fw - col_w, 0.0), float(fw)
        cx1, cx2 = int(round(cx1)), int(round(cx2))
        if cx2 <= cx1:
            return None
        crop = frame[y1:y2, cx1:cx2]
        if crop.size == 0:
            return None
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.in_w, self.in_h))
        if self.input_details[0]['dtype'] == np.int8:
            # 新 Re-ID 嵌入器(osnet/mobilenetv2)：归一化不在图内，Python 侧做 ImageNet
            # 归一化后按 input scale/zp 量化成 int8（uint8 分支是 Youtu/MobileNetV1，归一化烘焙进图内）
            iscale, izero = self.input_details[0]['quantization']
            norm_img = (resized.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
            q = np.clip(np.round(norm_img / iscale + izero), -128, 127).astype(np.int8)
            input_data = np.expand_dims(q, axis=0)
        else:
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
    下游 person_goal_bridge.py / objControl.py 零改动。

    轨迹状态机（SORT/ByteTrack 式 confirmed/lost + n_init，**维持低门槛 / 重捕高门槛**）：
      unlocked  —— 还没锁过。走"站稳3秒"自动锁 或 /Reid_Lock 手动锁。
      confirmed —— 正在跟。用 **sim_floor（低）** 维持，扛得住转身/侧身/距离变化。
                   连续 lost_grace 帧匹配不上 → lost。
      lost      —— 跟丢。**模板保留但停发 /Current_point**（下游据此停车+云台搜索）。
                   重捕要求 **relock_sim_floor（明显更高）** 且**连续 relock_min_hits 帧**都过线
                   才回 confirmed —— 防止目标不在时旁人勉强过线被认成目标（还被 EMA 坐实）。
                   试探帧一律不写回模板，避免污染。'''

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
        # 轨迹状态机（SORT/ByteTrack 式）：unlocked=还没锁过 / confirmed=正在跟(低门槛维持) /
        # lost=跟丢(模板保留、停发位置，重捕要过高门槛且连续多帧)
        self.state = "unlocked"
        self.miss_streak = 0            # confirmed 下连续匹配不上的帧数，达 lost_grace 判丢
        self.relock_hits = 0            # lost 下连续过高门槛的帧数，达 relock_min_hits 回 confirmed
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
        self.declare_parameter("sim_floor", 0.6)   # 外观相似度硬门槛。当前默认 reid_youtu(未剪枝 Youtu ResNet50)
        # 嵌入空间——沿用该模型历史定稿 0.6（离线贴地:目标中位0.73/旁人中位0.38/旁人p95~0.51，0.6 挡得住旁人独占）。
        # 注意躯干条裁剪几何已改过，**上车 VNC debug_sim 复核**。换模型必换此值(每个嵌入空间不同)。
        self.sim_floor = self.get_parameter('sim_floor').get_parameter_value().double_value
        self.declare_parameter("max_candidates", 3)   # 每次跑嵌入器最多处理的候选数上限，控制单帧TPU负载
        self.max_candidates = self.get_parameter('max_candidates').get_parameter_value().integer_value
        self.declare_parameter("ema_alpha", 0.1)   # 模板更新步长，只在高置信匹配时小步更新
        self.ema_alpha = self.get_parameter('ema_alpha').get_parameter_value().double_value
        self.declare_parameter("tie_margin", 0.05)   # 相似度差在此范围内视为并列，按空间连续性 tie-break
        self.tie_margin = self.get_parameter('tie_margin').get_parameter_value().double_value
        self.declare_parameter("ema_min_sim", 0.65)   # 只有匹配相似度明显高于 sim_floor 才更新模板，避免边界误匹配把模板带偏
        # （reid_youtu 历史定稿值，比 sim_floor 高一点留缓冲；上车 VNC 复核）
        self.ema_min_sim = self.get_parameter('ema_min_sim').get_parameter_value().double_value
        # 重锁迟滞（维持低/重捕高）：跟丢后不能再用 sim_floor 这个宽门槛去认人，否则目标不在时
        # 旁人只要勉强过线就被当成目标接上（还会被 EMA 坐实）。丢失后要求明显更高的相似度、
        # 且连续多帧都过，才允许重新锁上。参考 SORT/ByteTrack 的 confirmed/lost + n_init 门控。
        self.declare_parameter("relock_sim_floor", 0.75)   # 丢失后重捕的高门槛(必须明显>sim_floor)
        self.relock_sim_floor = self.get_parameter('relock_sim_floor').get_parameter_value().double_value
        self.declare_parameter("relock_min_hits", 5)       # 重捕需连续这么多帧都过高门槛才认
        self.relock_min_hits = self.get_parameter('relock_min_hits').get_parameter_value().integer_value
        self.declare_parameter("lost_grace", 5)            # confirmed 下连续这么多帧没匹配上→判丢失
        self.lost_grace = self.get_parameter('lost_grace').get_parameter_value().integer_value
        self.declare_parameter("debug_sim", False)   # 打印每个候选的相似度，供车上调参诊断
        self.debug_sim = self.get_parameter('debug_sim').get_parameter_value().bool_value
        self.declare_parameter("lock_min_size", 300)   # 自动初始锁定：候选框最长边(像素)要不小于这个值才算"离得够近"
        self.lock_min_size = self.get_parameter('lock_min_size').get_parameter_value().integer_value
        self.declare_parameter("lock_stable_time", 3.0)   # 上面那个候选框要持续满足条件这么多秒才真正锁定
        self.lock_stable_time = self.get_parameter('lock_stable_time').get_parameter_value().double_value
        self.declare_parameter("lock_stable_dist", 60.0)   # 候选框中心帧间移动不超过这个像素距离，才算"同一个人还站在原地"
        self.lock_stable_dist = self.get_parameter('lock_stable_dist').get_parameter_value().double_value

    def _enter_confirmed(self):
        '''初始锁定/手动换人/重锁成功后进入 confirmed，清计数。'''
        self.state = "confirmed"
        self.miss_streak = 0
        self.relock_hits = 0

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

    def _match(self, frame, candidates, floor):
        '''对所有候选跑嵌入器，取相似度过 floor 里最高的一个；
        相似度接近（tie_margin 内）时按离上一帧锁定框更近的 tie-break，减少帧间抖动。
        floor 由调用方按轨迹状态给：confirmed 用低门槛维持，lost 用高门槛重捕。'''
        best = None
        debug_sims = []
        for box, score in candidates:
            vec = self.embedder.embed(frame, box)
            if vec is None:
                continue
            sim = cosine_sim(vec, self.template)
            if self.debug_sim:
                debug_sims.append((box, sim))
            if sim < floor:
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
            self._enter_confirmed()
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
                    self._enter_confirmed()
                    box = biggest[0]
                    self.pending_lock_box = None
            else:
                self.pending_lock_box = biggest[0]   # 新出现的人/换了个人站过来，重新计时
                self.pending_lock_since = now
        elif self.template is not None:
            # 轨迹状态管理（SORT/ByteTrack 式 confirmed/lost + n_init 门控）：
            #   confirmed（正在跟）→ 用**低**门槛 sim_floor 维持，扛得住转身/侧身/距离变化；
            #   连续 lost_grace 帧匹配不上 → 判 lost（模板保留，但**停发** /Current_point）；
            #   lost 下重新捕获 → 用**明显更高**的 relock_sim_floor，且要连续 relock_min_hits 帧
            #   都过高门槛才回 confirmed —— 避免目标不在时把旁人误认成目标（趁虚而入）。
            floor = self.sim_floor if self.state == "confirmed" else self.relock_sim_floor
            best = self._match(frame, candidates, floor)
            if best is not None:
                cand_box, _, sim, vec = best
                self.last_box = cand_box
                if self.state == "lost":
                    self.relock_hits += 1
                    if self.relock_hits >= self.relock_min_hits:
                        self.state = "confirmed"      # 重锁成功
                        self.miss_streak = 0
                        self.relock_hits = 0
                        box = cand_box
                        print("[reid] relock CONFIRMED (sim=%.3f)" % sim)
                    # 还没攒够连续高置信帧：本帧不发布位置（下游据此继续停车/搜索）
                else:
                    self.miss_streak = 0
                    box = cand_box
                # 只在 confirmed 且相似度明显高于 sim_floor(ema_min_sim) 时才更新模板：
                # 边界匹配可用来发位置但不写回；lost 的试探帧一律不写回，防误匹配污染模板
                if self.state == "confirmed" and sim >= self.ema_min_sim:
                    self.template = (1 - self.ema_alpha) * self.template + self.ema_alpha * vec
                    norm = np.linalg.norm(self.template)
                    if norm > 1e-6:
                        self.template = self.template / norm
            else:
                self.relock_hits = 0    # 高置信必须**连续**，断一帧就重数
                if self.state == "confirmed":
                    self.miss_streak += 1
                    if self.miss_streak >= self.lost_grace:
                        self.state = "lost"
                        print("[reid] target LOST -> relock needs sim>=%.2f x%d frames"
                              % (self.relock_sim_floor, self.relock_min_hits))

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
            # 轨迹状态：lost 时显示重捕进度(连续过高门槛的帧数/需要的帧数)，方便标定门槛
            state_txt = self.state.upper()
            if self.state == "lost":
                state_txt += " relock %d/%d" % (self.relock_hits, self.relock_min_hits)
            cv.putText(frame, state_txt, (20, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                       (0, 255, 0) if self.state == "confirmed" else (0, 0, 255), 2)
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
