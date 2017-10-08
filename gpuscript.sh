
wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64-deb

sudo dpkg -i ./cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64-deb

sudo apt-key add /var/cuda-repo-9-0-local/7fa2af80.pub

sudo apt-get update 

sudo apt-get install cuda -y

sudo apt-get install python-pip -y

pip install --upgrade pip

pip install --upgrade tensorflow-gpu

sudo apt install nvidia-cuda-dev

sudo ldconfig /usr/local/cuda/lib64

pip install --user --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.10.0-cp27-none-linux_x86_64.whl
