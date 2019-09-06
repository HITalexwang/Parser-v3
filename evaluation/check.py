import sys

with open(sys.argv[1]) as f:
  sents = f.read().strip().split('\n\n')
  print len(sents)
  n_no_root = 0
  n_multi_root = 0
  n_no_head = 0
  n_self_circle = 0
  for sent in sents:
    n_root = 0
    lines = sent.strip().split('\n')
    for line in lines:
      line = line.strip().split('\t')
      if line[6] == '0':
        n_root += 1
      if line[0] == line[6]:
        n_self_circle += 1
      if line[6] == '_':
        n_no_head += 1
    if n_root == 0:
      n_no_root += 1
      #print sent
    elif n_root > 1:
      n_multi_root += 1
  print "No root:{}, Multi root:{}, No head:{}, Self circle:{}".format(n_no_root, n_multi_root, n_no_head, n_self_circle)
