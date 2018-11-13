main=/home/alex/work/codes/parser/Parser-v3/main.py
gpu=1,2
save=$1
conf=$2
if [[ -z $2 || $1 == '-h' ]];then
  echo "usage:./train.sh (save dir) (config file)"
  exit
fi
source /home/alex/work/env/py3/bin/activate
#CUDA_VISIBLE_DEVICES=$gpu python3 $main --save_dir $save train GraphMTLNetwork --config_file $conf --force --noscreen
CUDA_VISIBLE_DEVICES=$gpu python3 $main --save_dir $save train GraphStackedMTLNetwork --config_file $conf --force --noscreen
