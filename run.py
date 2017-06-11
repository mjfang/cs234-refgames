from collections import Counter

import tensorflow as tf
import numpy as np
import os
import reference_game
import cPickle as pickle
#load in data
dir = "data_processed"

#plan: sample concepts until all even...

def restrict_concepts(cats_to_concepts, max_size):
  for k in cats_to_concepts.keys():
    if len(cats_to_concepts[k]) > max_size:
      cats_to_concepts[k] = cats_to_concepts[k][:max_size]

files = os.listdir(dir)
concept_to_matrices = dict()
with open("cats_to_concepts","r") as f:
  cats_to_concepts = pickle.load(f)
# restrict_concepts(cats_to_concepts, 8)
concepts = set()
for key in cats_to_concepts.keys():
  concepts.update(cats_to_concepts[key])

#invert
concept_to_cat = dict()
for k in cats_to_concepts.keys():
  for c in cats_to_concepts[k]:
    concept_to_cat[c] = k

cat_count = Counter()
for file in files:
  file_path = os.path.join(dir, file)
  name = file.split(".")[0]
  if name in concepts:
    cat = concept_to_cat[name]
    if cat_count == 8:
      continue
    cat_count[cat] += 1
    concept_to_matrices[name] = np.load(file_path)
    concept_to_matrices[name] = (concept_to_matrices[name] - np.mean(concept_to_matrices[name], axis=0))
  # concept_to_matrices[file.split(".")[0]] = np.mean(concept_to_matrices[file.split(".")[0]], axis=1, keepdims=True)
  else:
    print( name, " not in concept_to_matrices")
  print("")
  # st = np.std(concept_to_matrices[file.split(".")[0]], axis=0)
  # st[np.where(st == 0)] = 1
  # concept_to_matrices[file.split(".")[0]] /= st



with tf.Session() as sess:
  np.random.seed(seed=1234)
  tf.set_random_seed(1234)
  reference_game.ReferenceGame(concept_to_cat).run(sess, concept_to_matrices, num_iters=50000)


