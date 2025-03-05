# 基于官方的 CUDA 11.1 基础镜像
FROM nvidia/cuda:11.1-base

# 使用清华大学的 apt 镜像源
RUN echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal main restricted universe multiverse" > /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-security main restricted universe multiverse" >> /etc/apt/sources.list

# 安装 curl 工具（用于下载 GPG 密钥）
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# 下载并添加 NVIDIA 的 GPG 密钥
RUN curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub | apt-key add -

# 配置 CUDA 和 NVIDIA ML 仓库
RUN echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

# 更新 apt 仓库并安装必要的系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    ca-certificates \
    curl \
    gnupg2 \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 3.8
RUN apt-get update && apt-get install -y python3.8 python3.8-dev python3-pip

# 设置 Python 3.8 为默认 Python
RUN ln -s /usr/bin/python3.8 /usr/bin/python

# 安装 pip 工具
RUN python -m pip install --upgrade pip

# 安装 PyTorch 1.9.0 和 CUDA Toolkit 11.1
RUN pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# 安装其他依赖
RUN pip install mmcv-full==1.6.0 mmdet==2.24.0 mmsegmentation==0.29.1 mmdet3d==1.0.0rc5 spconv-cu111==2.1.21 flash-attn==0.2.2

# 设置工作目录
WORKDIR /workspace

# 可选：复制本地代码到容器中
# COPY . /workspace

# 可选：设置默认命令
CMD ["bash"]