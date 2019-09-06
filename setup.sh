# Checking python version
PY_VER=none
if [ -x "$(command -v python3.5)" ];then
  PY_VER="python3.5"
fi
if [ -x "$(command -v python3.6)" ];then
  PY_VER="python3.6"
fi

if [ $PY_VER = none ];then
  echo "python3.5 is not installed, please install it first!"
  exit
else
  echo Using $PY_VER
fi

# Checking Virtualenv
if ! [ -x "$(command -v virtualenv)" ];then
  echo "virtualenv is not installed, trying to install it with pip."
  #pip install virtualenv
fi

# Building virtualenv
virtualenv env --python=$PY_VER
source env/bin/activate
pip install -r requirements.txt
