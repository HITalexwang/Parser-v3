main=main.py
save=$1
config=$2
log=$3
if [[ -z $2 || $1 == '-h' ]];then
  echo "usage:./run.sh (model dir) (config file) (log file)"
  exit
fi
#CUDA_VISIBLE_DEVICES=$gpu 
python3 $main --save_dir $save train GraphParserNetwork --config_file $config --force --noscreen > $log 2>&1