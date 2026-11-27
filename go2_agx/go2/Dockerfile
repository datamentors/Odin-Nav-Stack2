# Odin-Nav-Stack2 for the Go2 dock Jetson (JetPack 6 host, ROS1 inside the container).
# Image ons2:go2. Build from the REPOSITORY ROOT (COPY NeuPAN):  docker build -f go2_agx/go2/Dockerfile -t ons2:go2 .
# Base image odin-nav:drv0144 = docker/Dockerfile.global on dustynv/ros:noetic-pytorch-l4t-r35.3.1; go2_agx/build_images.sh builds it first.
# Base = our working ONS1 image (Ubuntu 20.04 + ROS Noetic + conda env "neupan" Python 3.10 + torch 2.1 + cvxpy/SCS).
# The ONS2 repo (~/ons2) is MOUNTED at the same path at runtime and built there (ros_ws/devel), so the official
# scripts (scripts/ons2_go2.sh ...) run unmodified paths. The old image odin-nav:drv0144 is untouched.
# The container ons2-go2 made from this image also runs the ROS master (start_odin_remote.sh) and ONS2 (ons2_start.sh).
FROM odin-nav:drv0144
RUN apt-get update && apt-get install -y --no-install-recommends \
      tmux ros-noetic-rosbridge-suite ros-noetic-foxglove-bridge \
    && rm -rf /var/lib/apt/lists/*
# ONS2 NeuPAN (analytic DUNE + SCS warm start) into this image only
COPY NeuPAN /opt/ons2_neupan
RUN /opt/conda/envs/neupan/bin/pip install --no-cache-dir "numpy<2" -e /opt/ons2_neupan
ENV NEUPAN_PYTHON=/opt/conda/envs/neupan/bin/python
# catkin builds the ONS2 workspace with the neupan Python (3.10) so every ROS node, NeuPAN included, runs on it
RUN /opt/conda/envs/neupan/bin/pip install --no-cache-dir "empy==3.3.4" catkin_pkg rospkg
# Official Unitree SDK2 (needed by ONS2 ros_ws/src/unitree_control, the Go2 chassis bridge). Installed in /usr/local
# INSIDE this image only (the host is untouched).
RUN apt-get update && apt-get install -y --no-install-recommends git cmake g++ build-essential libyaml-cpp-dev \
      libeigen3-dev libboost-all-dev libspdlog-dev libfmt-dev && rm -rf /var/lib/apt/lists/* \
 && git clone --depth 1 https://github.com/unitreerobotics/unitree_sdk2.git /opt/unitree_sdk2 \
 && cd /opt/unitree_sdk2 && mkdir build && cd build && cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local -DBUILD_EXAMPLES=OFF \
 && make -j6 install && ldconfig
# odin_vlm_terminal (ONS2 README 5.4) posts frames to an OpenAI-compatible VLM server with requests
RUN /opt/conda/envs/neupan/bin/pip install --no-cache-dir requests
# ONS2 yolo_ros (YOLOv5 submodule) dependencies in the neupan Python 3.10 that runs every ONS2 ROS node.
# Constraints keep the working torch 2.1.0 / numpy<2 (pip would otherwise replace them).
RUN printf "torch==2.1.0\nnumpy<2\n" > /tmp/c.txt && /opt/conda/envs/neupan/bin/pip install --no-cache-dir -c /tmp/c.txt \
      "opencv-python-headless<4.11" "torchvision==0.16.0" pandas seaborn tqdm psutil thop gitpython "ultralytics==8.3.40" "setuptools<70"
# Same versions as the first working image (a rebuild had pulled scs 3.3.2 and lacked netifaces). Last on purpose: cheap to change.
RUN /opt/conda/envs/neupan/bin/pip install --no-cache-dir "scs==3.2.11" "netifaces==0.11.0"
