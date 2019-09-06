from collections import defaultdict

root = 'root'
#===============================================================
def to_conllu(sdp_filename, conllu_filename):
  """"""
  
  with open(sdp_filename) as f:
    with open(conllu_filename, 'w') as g:
      buff = []
      sents = f.read().strip().split('\n\n')
      print ("sents: ", len(sents))
      for sent in sents:
        conllu_form = []
        words = []
        top = []
        preds = []
        graph = defaultdict(dict)
        lines = sent.strip().split('\n')
        for line in lines:
          if line.startswith('#'):
            conllu_form.append(line)
            continue
          items = line.strip().split('\t')
          words.append( [items[0], items[1], items[2], items[3], items[3], '_'] )
          if items[4] is '+':
            top.append(int(items[0])-1)
          if items[5] is '+':
            preds.append(int(items[0]))
          for j, p in enumerate(items[7:]):
            graph[int(items[0])-1][j] = p
        for i, word in enumerate(words):
          augment = []
          if i in top:
            word.extend(['0', root])
            augment.append('0:'+root)
          for j, p in graph[i].items():
            if graph[i][j] is not '_':
              head = preds[j]
              if len(word) == 6:
                word.extend([str(head), p])
              augment.append(str(head)+':'+p)
          if len(word) == 6:
            word.extend(['_', '_'])
          augment = '|'.join(augment)
          if augment:
            word.extend([augment, '_'])
          else:
            word.extend(['_', '_'])
          conllu_form.append('\t'.join(word))
        g.write('\n'.join(conllu_form) + '\n\n')

#***************************************************************
if __name__ == '__main__':
  """"""
  
  import sys
  to_conllu(sys.argv[1], sys.argv[2])
