FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel

# Install some basic utilities
COPY apt-sources.list /etc/apt/sources.list
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    gcc \
    g++ \
    libusb-1.0-0 \
    libgl1-mesa-glx \
    libglib2.0-dev \
    openssh-server \
    openssh-client \
    iputils-ping \
    unzip \
    cmake \
    libssl-dev \
    libosmesa6-dev \
    freeglut3-dev \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# Rebuild opengl 
RUN git clone https://github.com/mmatl/pyopengl.git
RUN pip install ./pyopengl --user

# For the convenience for users in China mainland
RUN pip config set global.index-url https://mirrors.cloud.tencent.com/pypi/simple
COPY requirements.txt requirements.txt 
RUN pip install -r requirements.txt --user

# Install KNN_CUDA
COPY KNN_CUDA-0.2-py3-none-any.whl KNN_CUDA-0.2-py3-none-any.whl
RUN pip install KNN_CUDA-0.2-py3-none-any.whl --user

CMD ["python3"]

