#!/bin/bash
xhost +
docker run -it --rm \
--privileged=true \
--net=host \
--env="DISPLAY" \
--env="QT_X11_NO_MITSHM=1" \
-v /tem/.X11-unix:/tmp/.X11-unix \
--security-opt apparmor:unconfined \
-v /dev/input:/dev/input \
-v /dev/video0:/dev/video0 \
yahboomtechnology/ros-humble:4.0.3 /bin/bash /root/1.sh 

