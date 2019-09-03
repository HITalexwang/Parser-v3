main=main.py
save=$1
out=$2
file=$3
other=$4
if [[ -z $2 || $1 == '-h' ]];then
  echo "usage:./run.sh (model dir) (output filename) (input filename) (other model dirs)"
  exit
fi
#CUDA_VISIBLE_DEVICES=$gpu 
python3 $main --save_dir $save run --output_filename $out $file --other_save_dirs $other
