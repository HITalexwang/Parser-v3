# Predicting
#./run.sh data/text-conll/text.test.conll outputs
./run.sh data/text-conll/text.test-input.conll outputs

# Evaluating
#./eval.sh data/text-conll/text.test.conll outputs/text.test.conll.conllu
./eval.sh data/text-conll/text.test.conll outputs/text.test-input.conll.conllu.sem16
