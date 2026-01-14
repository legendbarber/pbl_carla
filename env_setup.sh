#!/bin/bash

# locale setup
locale
sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
locale

# ros2 apt repository setup
sudo apt install software-properties-common -y
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}')
curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb"
sudo dpkg -i /tmp/ros2-apt-source.deb

# ros2 package installation
sudo apt update && sudo apt upgrade -y
sudo apt install ros-humble-desktop -y
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc

cd ~            
mkdir -p carla  
cd ~/carla      
#
# Download manually CARLA 0.9.16 (https://github.com/carla-simulator/carla/releases/tag/0.9.16/)
#
tar -xvzf CARLA_0.9.16.tar.gz
# Set up CARLA Python API
sudo apt update 
sudo apt install -y build-essential g++-12 cmake ninja-build libvulkan1 python3 python3-dev python3-pip python3-venv autoconf wget curl rsync unzip git git-lfs libpng-dev libtiff5-dev libjpeg-dev
pip3 install ~/carla/PythonAPI/carla/dist/carla-0.9.16-cp310-cp310-manylinux_2_31_x86_64.whl