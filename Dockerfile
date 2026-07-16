# Start from the official ROS 2 Humble Desktop image
FROM osrf/ros:humble-desktop

# Set environment variables to avoid user prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies (including Orbbec USB requirements and OpenCV dependencies)
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    git \
    build-essential \
    ninja-build \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libusb-1.0-0-dev \
    libudev-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /workspace/FlowCloth

# Copy the entire repository into the container
COPY . /workspace/FlowCloth

# Create a Python virtual environment WITH system packages enabled (critical for inheriting ROS 2's rclpy)
ENV VIRTUAL_ENV=/workspace/FlowCloth/.venv
RUN python3 -m venv $VIRTUAL_ENV --system-site-packages

# Add the virtual environment to the system PATH so pip/python run inside it automatically
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# 1. Install the custom diffusers fork used by UniClothDiff
RUN cd third_party/diffusers && pip install -e .

# 2. Install torkit3d (required by UniClothDiff)
RUN pip install "git+https://github.com/Jiayuan-Gu/torkit3d.git"

# 3. Install the rest of the dependencies and FlowCloth itself
# (This uses the setup.py and environment details in the repo)
RUN pip install -e .

# Automatically source ROS 2 and the Python Virtual Environment when the container starts
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
RUN echo "source /workspace/FlowCloth/.venv/bin/activate" >> ~/.bashrc

# Default command when spinning up the container
CMD ["/bin/bash"]
