tosem16=converter/conllu_to_sem16.py
#check=evaluation/check.py
eval=evaluation/evaluate.py

gold=$1 #data/text-conll/text.test.conll
pred=$2 #outputs/text.test.conllu

if [[ -z $1 || $1 == '-h' ]];then
  echo "usage:./eval.sh gold-file pred-file"
  exit
fi

#python $tosem16 $pred
#python $check $pred.sem16
python $eval --reference $gold --answer $pred --language chen2014ch
