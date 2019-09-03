eval=examples/evaluate_conllu.py
gold=$1
pred=$2
if [[ -z $2 || $1 == '-h' ]];then
  echo "params: (gold file) (predicted file)"
  exit
fi

python $eval --reference $gold --answer $pred --language chen2014ch
