#!/usr/bin/env bash

# 参数:
python_version="3.8.10";
system_version="centos";


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

echo "python_version: ${python_version}";
echo "system_version: ${system_version}";


if [ ${system_version} = "centos" ]; then
  # 安装 python 开发编译环境
  yum -y groupinstall "Development tools"
  yum -y install zlib-devel bzip2-devel openssl-devel ncurses-devel sqlite-devel readline-devel tk-devel gdbm-devel db4-devel libpcap-devel xz-devel
  yum install libffi-devel -y
  yum install -y wget
  yum install -y make

  mkdir /data/dep
  wget -P /data/dep https://www.python.org/ftp/python/${python_version}/Python-${python_version}.tgz

  cd /data/dep || exit 1;
  tar -zxvf Python-${python_version}.tgz
  cd /data/dep/Python-${python_version} || exit 1;

  mkdir /usr/local/python-${python_version}
  ./configure --prefix=/usr/local/python-${python_version}
  make && make install

  /usr/local/python-${python_version}/bin/python3 -V
  /usr/local/python-${python_version}/bin/pip3 -V

  rm -rf /usr/local/bin/python3
  rm -rf /usr/local/bin/pip3
  ln -s /usr/local/python-${python_version}/bin/python3 /usr/local/bin/python3
  ln -s /usr/local/python-${python_version}/bin/pip3 /usr/local/bin/pip3

  python3 -V
  pip3 -V

elif [ ${system_version} = "ubuntu" ]; then
  # 安装 python 开发编译环境
  # https://zhuanlan.zhihu.com/p/506491209

  # 刷新软件包目录
  sudo apt update
  # 列出当前可用的更新
  sudo apt list --upgradable
  # 如上一步提示有可以更新的项目，则执行更新
  sudo apt -y upgrade
  # 安装 GCC 编译器
  sudo apt install gcc
  # 检查安装是否成功
  gcc -v

  # 安装依赖
  sudo apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libbz2-dev liblzma-dev sqlite3 libsqlite3-dev tk-dev uuid-dev libgdbm-compat-dev

  mkdir /data/dep

  # sudo wget -P /data/dep https://www.python.org/ftp/python/3.6.5/Python-3.6.5.tgz
  sudo wget -P /data/dep https://www.python.org/ftp/python/${python_version}/Python-${python_version}.tgz

  cd /data/dep || exit 1;
  # tar -zxvf Python-3.6.5.tgz
  tar -zxvf Python-${python_version}.tgz
  # cd /data/dep/Python-3.6.5
  cd /data/dep/Python-${python_version} || exit 1;
  # mkdir /usr/local/python-3.6.5
  mkdir /usr/local/python-${python_version}

  # 检查依赖与配置编译
  # sudo ./configure --prefix=/usr/local/python-3.6.5 --enable-optimizations --with-lto --enable-shared
  sudo ./configure --prefix=/usr/local/python-${python_version} --enable-optimizations --with-lto --enable-shared
  cpu_count=$(cat /proc/cpuinfo | grep processor | wc -l)
  # sudo make -j 4
  sudo make -j "${cpu_count}"

  /usr/local/python-${python_version}/bin/python3 -V
  /usr/local/python-${python_version}/bin/pip3 -V

  rm -rf /usr/local/bin/python3
  rm -rf /usr/local/bin/pip3
  ln -s /usr/local/python-${python_version}/bin/python3 /usr/local/bin/python3
  ln -s /usr/local/python-${python_version}/bin/pip3 /usr/local/bin/pip3

  python3 -V
  pip3 -V
fi
