# syntax=docker/dockerfile:1

FROM rocm/dev-ubuntu-24.04:6.4 AS base

FROM base AS devcontainer

ARG USERNAME=vscode
ARG USER_UID=1001
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME -s /usr/bin/bash \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

RUN apt-get update \
    && apt-get install -y git python3-dev python-is-python3 python3-venv \
    && pip install --break-system-packages pre-commit black ruff

USER $USERNAME
