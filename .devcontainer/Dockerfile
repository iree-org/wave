ARG ROCM_VERSION=7.2
ARG UBUNTU_VERSION=24.04
FROM rocm/dev-ubuntu-${UBUNTU_VERSION}:${ROCM_VERSION}-complete

ENV DEBIAN_FRONTEND=noninteractive

ARG LLVM_VERSION=20
ARG PYTHON_VERSION=3.12
ARG UBUNTU_VERSION=24.04

RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    ca-certificates \
    software-properties-common \
    git \
    zsh \
    tmux \
    sudo \
    vim \
    ccache \
    ninja-build \
    cmake \
    && apt-get install -y \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
    # Clean up apt cache to reduce image size.
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p -m 755 /etc/apt/keyrings \
    && curl -nv -o /etc/apt/keyrings/githubcli-archive-keyring.gpg https://cli.github.com/packages/githubcli-archive-keyring.gpg \
    && chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
    && mkdir -p -m 755 /etc/apt/sources.list.d \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" > /etc/apt/sources.list.d/github-cli.list \
    && apt-get update \
    && apt-get install -y gh \
    && rm -rf /var/lib/apt/lists/*

RUN CODENAME=$([ "${UBUNTU_VERSION}" = "22.04" ] && echo "jammy" || echo "noble") && \
    curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key | gpg --dearmor -o /etc/apt/keyrings/llvm.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/llvm.gpg] https://apt.llvm.org/${CODENAME}/ llvm-toolchain-${CODENAME}-${LLVM_VERSION} main" \
        > /etc/apt/sources.list.d/llvm.list && \
    apt-get update && apt-get install -y \
        clang-${LLVM_VERSION} \
        clang++-${LLVM_VERSION} \
        lld-${LLVM_VERSION} \
        lldb-${LLVM_VERSION} \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 100 && \
    update-alternatives --install /usr/bin/python  python  /usr/bin/python${PYTHON_VERSION} 100 && \
    update-alternatives --install /usr/bin/clang   clang   /usr/bin/clang-${LLVM_VERSION}   100 && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-${LLVM_VERSION} 100 && \
    update-alternatives --install /usr/bin/lld     lld     /usr/bin/lld-${LLVM_VERSION}     100 && \
    update-alternatives --install /usr/bin/lldb    lldb    /usr/bin/lldb-${LLVM_VERSION}    100

RUN id ubuntu >/dev/null 2>&1 || useradd -m -s /bin/bash -U ubuntu && \
    usermod -s /usr/bin/zsh ubuntu && \
    usermod -aG render,video ubuntu && \
    echo "ubuntu ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/ubuntu && \
    chmod 0440 /etc/sudoers.d/ubuntu

USER ubuntu
WORKDIR /home/ubuntu

RUN curl -fsSL https://claude.ai/install.sh | bash && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
