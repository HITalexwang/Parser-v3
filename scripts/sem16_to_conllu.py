from collections import defaultdict
import codecs

root = 'root'
#===============================================================
def to_conllu(sdp_filename, conllu_filename):
  """"""

  with codecs.open(sdp_filename, 'r', "utf-8") as f:
    with open(conllu_filename, 'w') as g:
      buff = []
      sents = f.read().strip().split('\n\n')
      print (sents)
      print ("sents: ", len(sents))
      for sent in sents:
        #print (sent)
        conllu_form = []
        words = []
        lines = sent.strip().split('\n')
        for line in lines:
          if line.startswith('#'):
            conllu_form.append(line)
            continue
          items = line.strip().split('\t')
          if int(items[0]) == len(words) + 1:
            if items[6] is not '_' and items[7] is not '_':
              items[8] = items[6]+':'+items[7]
            words.append(items)
          elif int(items[0]) == len(words):
            words[-1][8] += '|'+items[6]+':'+items[7]
          #else:
          #  print "Error:{}".format(line)
        for word in words:
          conllu_form.append('\t'.join(word))
        g.write('\n'.join(conllu_form).encode("utf-8") + '\n\n')

#***************************************************************
if __name__ == '__main__':
  """"""

  import sys
  to_conllu(sys.argv[1], sys.argv[2])
