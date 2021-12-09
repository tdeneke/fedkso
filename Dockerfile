# FROM python:3.8.9
# Start FROM Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
# Adopted from KSO: https://github.com/ocean-data-factory-sweden/koster_yolov4

FROM nvcr.io/nvidia/pytorch:21.05-py3

# Install linux packages
RUN apt update && apt install -y zip htop screen libgl1-mesa-glx

COPY requirements.txt /app/
RUN mkdir -p /app/logs/wandb/
WORKDIR /app

RUN python -m pip install --upgrade pip \
    && pip uninstall -y nvidia-tensorboard nvidia-tensorboard-plugin-dlprof \
    && pip install --no-cache -r requirements.txt coremltools onnx gsutil notebook \
    && pip install --no-cache -U torch torchvision numpy \
    # RUN pip install -r requirements.txt
    && pip install --no-cache -e git://github.com/scaleoutsystems/fedn.git@develop#egg=fedn\&subdirectory=fedn

# Set environment variables
ENV HOME=/app
ENV WANDB_DIR=/app/logs/wandb/
ENV WANDB_CACHE_DIR=/app/logs/wandb/
