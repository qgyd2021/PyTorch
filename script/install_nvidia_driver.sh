#!/usr/bin/env bash
#GPU驱动安装需要先将原有的显示关闭, 重启机器, 再进行安装.
#参考链接:
#https://blog.csdn.net/kingschan/article/details/19033595
#https://blog.csdn.net/HaixWang/article/details/90408538
#
#>>> yum install -y pciutils
#查看 linux 机器上是否有 GPU
#lspci |grep -i nvidia
#
#>>> lspci |grep -i nvidia
#00:08.0 3D controller: NVIDIA Corporation TU104GL [Tesla T4] (rev a1)
#
#
#NVIDIA 驱动程序下载
#先在 pytorch 上查看应该用什么 cuda 版本, 再安装对应的 cuda-toolkit cuda.
#再根据 gpu 版本下载安装对应的 nvidia 驱动
#
## pytorch 版本
#https://pytorch.org/get-started/locally/
#
## CUDA 下载 (好像不需要这个)
#https://developer.nvidia.com/cuda-toolkit-archive
#
## nvidia 驱动
#https://www.nvidia.cn/Download/index.aspx?lang=cn
#http://www.nvidia.com/Download/index.aspx
#
#在下方的下拉列表中进行选择，针对您的 NVIDIA 产品确定合适的驱动。
#产品类型:
#Data Center / Tesla
#产品系列:
#T-Series
#产品家族:
#Tesla T4
#操作系统:
#Linux 64-bit
#CUDA Toolkit:
#10.2
#语言:
#Chinese (Simpleified)
#
#
#>>> mkdir -p /data/tianxing
#>>> cd /data/tianxing
#>>> wget https://cn.download.nvidia.com/tesla/440.118.02/NVIDIA-Linux-x86_64-440.118.02.run
#>>> sh NVIDIA-Linux-x86_64-440.118.02.run
#
## 异常:
#ERROR: The Nouveau kernel driver is currently in use by your system.  This driver is incompatible with the NVIDIA driver, and must be disabled before proceeding.  Please consult the NVIDIA driver README and your
#Linux distribution's documentation for details on how to correctly disable the Nouveau kernel driver.
#[OK]
#
#For some distributions, Nouveau can be disabled by adding a file in the modprobe configuration directory.  Would you like nvidia-installer to attempt to create this modprobe file for you?
#[NO]
#
#ERROR: Installation has failed.  Please see the file '/var/log/nvidia-installer.log' for details.  You may find suggestions on fixing installation problems in the README available on the Linux driver download
#page at www.nvidia.com.
#[OK]
#
## 参考链接:
#https://blog.csdn.net/kingschan/article/details/19033595
#
## 禁用原有的显卡驱动 nouveau
#>>> echo -e "blacklist nouveau\noptions nouveau modeset=0\n" > /etc/modprobe.d/blacklist-nouveau.conf
#>>> sudo dracut --force
## 重启
#>>> reboot
#
#>>> init 3
#>>> sh NVIDIA-Linux-x86_64-440.118.02.run
#
## 异常
#ERROR: Unable to find the kernel source tree for the currently running kernel. Please make sure you have installed the kernel source files for your kernel and that they are properly configured; on Red Hat Linux systems, for example, be sure you have the 'kernel-source' or 'kernel-devel' RPM installed. If you know the correct kernel source files are installed, you may specify the kernel source path with the '--kernel-source-path' command line option.
#[OK]
#ERROR: Installation has failed.  Please see the file '/var/log/nvidia-installer.log' for details.  You may find suggestions on fixing installation problems in the README available on the Linux driver download
#page at www.nvidia.com.
#[OK]
#
## 参考链接
## https://blog.csdn.net/HaixWang/article/details/90408538
#
#>>> uname -r
#3.10.0-1160.49.1.el7.x86_64
#>>> yum install kernel-devel kernel-headers -y
#>>> yum info kernel-devel kernel-headers
#>>> yum install -y "kernel-devel-uname-r == $(uname -r)"
#>>> yum -y distro-sync
#
#>>> sh NVIDIA-Linux-x86_64-440.118.02.run
#
## 安装成功
#WARNING: nvidia-installer was forced to guess the X library path '/usr/lib64' and X module path '/usr/lib64/xorg/modules'; these paths were not queryable from the system.  If X fails to find the NVIDIA X driver
#module, please install the `pkg-config` utility and the X.Org SDK/development package for your distribution and reinstall the driver.
#[OK]
#Install NVIDIA's 32-bit compatibility libraries?
#[YES]
#Installation of the kernel module for the NVIDIA Accelerated Graphics Driver for Linux-x86_64 (version 440.118.02) is now complete.
#[OK]
#
#
## 查看 GPU 使用情况; watch -n 1 -d nvidia-smi 每1秒刷新一次.
#>>> nvidia-smi
#Thu Mar  9 12:00:37 2023
#+-----------------------------------------------------------------------------+
#| NVIDIA-SMI 440.118.02   Driver Version: 440.118.02   CUDA Version: 10.2     |
#|-------------------------------+----------------------+----------------------+
#| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
#| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
#|===============================+======================+======================|
#|   0  Tesla T4            Off  | 00000000:00:08.0 Off |                  Off |
#| N/A   54C    P0    22W /  70W |      0MiB / 16127MiB |      0%      Default |
#+-------------------------------+----------------------+----------------------+
#
#+-----------------------------------------------------------------------------+
#| Processes:                                                       GPU Memory |
#|  GPU       PID   Type   Process name                             Usage      |
#|=============================================================================|
#|  No running processes found                                                 |
#+-----------------------------------------------------------------------------+
#
#

# params
stage=1
nvidia_driver_filename=https://cn.download.nvidia.com/tesla/440.118.02/NVIDIA-Linux-x86_64-440.118.02.run

# parse options
while true; do
  [ -z "${1:-}" ] && break;  # break if there are no arguments
  case "$1" in
    --*) name=$(echo "$1" | sed s/^--// | sed s/-/_/g);
      eval '[ -z "${'"$name"'+xxx}" ]' && echo "$0: invalid option $1" 1>&2 && exit 1;
      old_value="(eval echo \\$$name)";
      if [ "${old_value}" == "true" ] || [ "${old_value}" == "false" ]; then
        was_bool=true;
      else
        was_bool=false;
      fi

      # Set the variable to the right value-- the escaped quotes make it work if
      # the option had spaces, like --cmd "queue.pl -sync y"
      eval "${name}=\"$2\"";

      # Check that Boolean-valued arguments are really Boolean.
      if $was_bool && [[ "$2" != "true" && "$2" != "false" ]]; then
        echo "$0: expected \"true\" or \"false\": $1 $2" 1>&2
        exit 1;
      fi
      shift 2;
      ;;

    *) break;
  esac
done

echo "stage: ${stage}";

yum -y install wget
yum -y install sudo

if [ ${stage} -eq 0 ]; then
  mkdir -p /data/dep
  cd /data/dep || echo 1;
  wget -P /data/dep ${nvidia_driver_filename}

  echo -e "blacklist nouveau\noptions nouveau modeset=0\n" > /etc/modprobe.d/blacklist-nouveau.conf
  sudo dracut --force
  # 重启
  reboot
elif [ ${stage} -eq 1 ]; then
  init 3

  yum install -y kernel-devel kernel-headers
  yum info kernel-devel kernel-headers
  yum install -y "kernel-devel-uname-r == $(uname -r)"
  yum -y distro-sync

  cd /data/dep || echo 1;

  # 安装时, 需要回车三下.
  sh NVIDIA-Linux-x86_64-440.118.02.run
  nvidia-smi
fi
