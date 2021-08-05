This is a CloudLab profile (and associated scripts and other utilities) for experiments involving offloading computer vision tasks from a wearable device to a cloud server.

## Using this repository

The contents of this repository should be cloned into the directory `/local/repository` on all nodes in the experiment.

```
git clone https://github.com/ffund/offload-cloudlab/ /local/repository
```

## CloudLab profiles

To get four nodes in a line, with a GPU on the server node, use [offload](https://www.cloudlab.us/p/nyunetworks/offload).

To get a single server node with a GPU, use [wearable](https://www.cloudlab.us/p/nyunetworks/wearable). 

## Setting up the server

### CUDA and ZED SDK

The disk image on the server node already has CUDA 11.0 and ZED SDK 3.5.0 installed.

### Darknet

First, expand the root filesystem to 40GB so that it will have enough space for all the libraries etc.

```
sudo su
chmod a+x /local/repository/grow-rootfs.sh
env RESIZEROOT=40 /local/repository/grow-rootfs.sh
reboot
```

Wait for the node to reboot, then log in again. Verify that the root disk is expanded to 40G, e.g.:

```
ffund00@server:~$ df -h
Filesystem                                  Size  Used Avail Use% Mounted on
udev                                         94G     0   94G   0% /dev
tmpfs                                        19G  1.7M   19G   1% /run
/dev/sda1                                    40G   14G   25G  36% /
tmpfs                                        94G     0   94G   0% /dev/shm
tmpfs                                       5.0M     0  5.0M   0% /run/lock
tmpfs                                        94G     0   94G   0% /sys/fs/cgroup
/dev/mapper/emulab-bs2                       92G  106M   87G   1% /data
ops.wisc.cloudlab.us:/share                  50G  2.0G   49G   4% /share
ops.wisc.cloudlab.us:/proj/nyunetworks-PG0  100G   27G   74G  27% /proj/nyunetworks-PG0
tmpfs                                        19G     0   19G   0% /run/user/20001
```

Next, we will set up the extra disk space at `/data`:

```
sudo chown $USER /data
```

Now, install some pre-requisites for Darknet.

Install OpenCV and ZED Python libraries:

```
sudo pip3 install --upgrade pip
sudo pip3 install opencv-python
sudo python3 -m pip install --ignore-installed /usr/local/zed/pyzed-3.5-cp36-cp36m-linux_x86_64.whl
```

Install a newer `cmake`:

```
cd /data
wget https://github.com/Kitware/CMake/releases/download/v3.21.1/cmake-3.21.1.tar.gz
tar -xzvf cmake-3.21.1.tar.gz 
cd cmake-3.21.1/
./bootstrap
make -j$(nproc)
sudo make install
```

Install cudnn:

```
cd /data
wget http://people.cs.uchicago.edu/~kauffman/nvidia/cudnn/cudnn-11.0-linux-x64-v8.2.0.53.tgz
tar -xvzf cudnn-11.0-linux-x64-v8.2.0.53.tgz
 
sudo cp cuda/include/cudnn*.h /usr/local/cuda/include 
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64 
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

Now, install Darknet:

```
cd /data
git clone https://github.com/AlexeyAB/darknet
cd darknet
mkdir build_release
cd build_release

export CUDA_PATH=/usr/local/cuda-11.0/bin
export CUDACXX=/usr/local/cuda-11.0/bin/nvcc
cmake ..
cmake --build . --target install --parallel 8
```

Download the YOLO weights:

```
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights -O /data/yolov4.weights
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov3.weights -O /data/yolov3.weights
```

Create a COCO config file in `/data/coco.data` with the following contents:

```
classes= 80
names = /data/darknet/cfg/coco.names
eval=coco
```

## Running the distance estimation script

Transfer one `.svo` file to the `/data` directory on the server.

The script expects a TCP listener on port 9998, so in a second terminal on the server, run

```
netcat -l 9998
```

Then you can run the script with

```
DARKNET='/data/darknet/' python3 /local/repository/darknet_zed_tcp.py -s /data/89_07062021_LCS_LG_HD1080_30_12000_Jacky.svo -m /data/coco.data -w /data/yolov3.weights
```
