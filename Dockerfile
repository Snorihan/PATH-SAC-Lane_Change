# Use an official PyTorch image as a parent image.
# This includes Python, CUDA, and cuDNN, suitable for deep learning tasks.
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Set the working directory in the container
WORKDIR /app

# Install system-level dependencies required for rendering and video processing.
# - git: For installing packages from git repositories.
# - xvfb: A virtual framebuffer for running graphical applications (like the env renderer) without a display.
# - ffmpeg: For recording videos of the environment.
# - libgl1-mesa-glx: OpenGL library required by pygame for rendering.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    xvfb \
    ffmpeg \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker layer caching.
# This layer is only rebuilt if requirements.txt changes.
COPY requirements.txt .

# Install Python dependencies.
# Note: torch is already included in the base pytorch/pytorch image.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project source code into the container.
COPY . .

# Add the project root to PYTHONPATH to ensure modules can be imported correctly.
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Set the default command to run the SAC training script.
# `xvfb-run` is used to provide a virtual display, which is often required by
# rendering libraries like pygame, even for "rgb_array" mode.
CMD ["xvfb-run", "-a", "python", "sac/thirdparty_cleanrl/sac_core.py", "--env-id", "lane-changing-v0"]