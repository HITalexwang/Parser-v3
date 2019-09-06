from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

#===============================================================
def conllu_to_sem16(conllu_filename):
  """"""

  with open(conllu_filename) as f:
    with open(conllu_filename+'.sem16', 'w') as g:
      buff = []
      for line in f:
        line = line.strip('\n')
        items = line.split('\t')
        if len(items) == 10:
          # Add it to the buffer
          buff.append(items)
        elif buff:
          #g.write('#\n')
          # Process the buffer
          for i, items in enumerate(buff):
            #words.append(line)
            if items[8] != '_':
              nodes = items[8].split('|')
              for node in nodes:
                words = items
                # copy xpos to upos
                words[3] = words[4]
                node = node.split(':', 1)
                node[0] = int(node[0])
                words[6], words[7], words[8] = str(node[0]), node[1], '_'
                #words[7] = node[1]
                #words[8] = '_'
                g.write('\t'.join(words) + '\n')
            else:
              g.write('\t'.join(items) + '\n')
          g.write('\n')
          buff = []

#***************************************************************
if __name__ == '__main__':
  """"""

  import sys
  conllu_to_sem16(sys.argv[1])

