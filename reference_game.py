import numpy as np
import agents
import tensorflow as tf
import datetime
import os
import cPickle as pickle
from collections import Counter
import matplotlib.pyplot as plt

import agents3


class ReferenceGame:

  #This can be faster...
  # return x_target (batch_size, in_size, 2) permutation (batch size)

  def sample_exclude(self, end, exclude):
    if end <= 1:
      raise Exception('end has to be > 1!')
    ret = np.random.randint(end)
    while ret == exclude:
      ret = np.random.randint(end)
    return ret

  def get_batch_replace_object(self, batch_size, concept_to_matrices, in_size=4096):
    batch_record = []
    batch = np.zeros((batch_size, in_size, 2))
    concepts = concept_to_matrices.keys()
    num_concepts = len(concepts)
    for i in range(batch_size):
      #sample two concepts
      c_idx = np.random.randint(num_concepts)
      c_idx2 = self.sample_exclude(num_concepts, c_idx)
      c1 = concepts[c_idx]
      c2 = concepts[c_idx2]

      mat1 = concept_to_matrices[c1]
      num_im = mat1.shape[0]
      idx = np.random.randint(num_im)
      im1 = mat1[idx]

      mat2 = concept_to_matrices[c2]
      num_im = mat2.shape[0]
      idx2 = np.random.randint(num_im)
      im2 = mat2[idx2]
      batch[i, :, 0] = im1
      batch[i, :, 1] = im2
      batch_record.append((c1, c2, idx, idx2)) #concept 1 and 2, idx 1 and 2

    permute = np.zeros((2, batch_size), dtype=np.int8)
    # permute[1, :] = 1
    permute[np.random.randint(2, size=batch_size), np.arange(batch_size)] = 1
    return batch, permute, batch_record




  def get_batch(self, batch_size, concept_to_matrices, in_size=4096):
    batch_record = []
    batch = np.zeros((batch_size, in_size, 2))
    concepts = concept_to_matrices.keys()
    num_concepts = len(concepts)
    for i in range(batch_size):
      #sample two concepts
      c_idx = np.random.randint(num_concepts)
      c_idx2 = self.sample_exclude(num_concepts, c_idx)
      c1 = concepts[c_idx]
      c2 = concepts[c_idx2]

      mat1 = concept_to_matrices[c1]
      num_im = mat1.shape[0]
      idx = np.random.randint(num_im)
      im1 = mat1[idx]

      mat2 = concept_to_matrices[c2]
      num_im = mat2.shape[0]
      idx2 = np.random.randint(num_im)
      im2 = mat2[idx2]
      batch[i, :, 0] = im1
      batch[i, :, 1] = im2
      batch_record.append((c1, c2, idx, idx2)) #concept 1 and 2, idx 1 and 2

    permute = np.zeros((2, batch_size), dtype=np.int8)
    # permute[1, :] = 1
    permute[np.random.randint(2, size=batch_size), np.arange(batch_size)] = 1
    return batch, permute, batch_record

  def get_batchN(self, batch_size, concept_to_matrices, in_size=4096):
    batch_record = []
    batch = np.zeros((batch_size, in_size, 3))
    concepts = concept_to_matrices.keys()
    num_concepts = len(concepts)
    for i in range(batch_size):
      #sample three concepts
      c_idx = np.random.randint(num_concepts)
      c_idx2 = self.sample_exclude(num_concepts, c_idx)
      c_idx3 = self.sample_exclude(num_concepts, c_idx)

      c1 = concepts[c_idx]
      c2 = concepts[c_idx2]
      c3 = concepts[c_idx3]

      mat1 = concept_to_matrices[c1]
      num_im = mat1.shape[0]
      idx = np.random.randint(num_im)
      im1 = mat1[idx]

      mat2 = concept_to_matrices[c2]
      num_im = mat2.shape[0]
      idx2 = np.random.randint(num_im)
      im2 = mat2[idx2]

      mat3 = concept_to_matrices[c3]
      num_im = mat3.shape[0]
      idx3 = np.random.randint(num_im)
      im3 = mat3[idx3]

      batch[i, :, 0] = im1
      batch[i, :, 1] = im2
      batch[i, :, 2] = im3
      batch_record.append((c1, c2, c3, idx, idx2, idx3)) #concept 1 and 2, idx 1 and 2

    permute = np.zeros((3, batch_size), dtype=np.int8)
    # permute[1, :] = 1
    rand = np.random.randint(3, size=batch_size)
    permute[rand, np.arange(batch_size)] = 1
    for i in range(batch_size):
      permute[self.sample_exclude(3,rand[i]), i] = 2

    return batch, permute, batch_record



  def get_random_purity(self, batch_record, symbols):
    return self.get_purity(batch_record, np.random.permutation(symbols))


  def get_purity(self, batch_record, symbols):
    #1) identify the categories that are used with each
    symbol_cats = dict()

    for idx in range(len(symbols)):
      sym = symbols[idx]
      cat = self.concept_to_cat[batch_record[idx][0]]
      if sym not in symbol_cats.keys():
        symbol_cats[sym] = Counter()
      symbol_cats[sym][cat] += 1
    c = Counter()
    for k in symbol_cats.keys():
      c += symbol_cats[k]
    #2 For each symbol, calculate purity

    total = 0
    maj = 0
    for sym in symbol_cats.keys():
      majority = symbol_cats[sym].most_common(1)[0]
      total_num = sum(symbol_cats[sym].values())
      total += total_num
      maj += majority[1]
    print maj / float(total)
  def get_mistakes(self, reward, batch_record):
    print(len(np.where(reward[0] == 0)))
    for idx in np.where(reward == 0)[0]:
      c1, c2, idx1, idx2 = batch_record[idx]
      try:
        print c1, self.idx_to_path[(c1, idx1)], c2, self.idx_to_path[(c2, idx2)]
      except Exception, e:
        print ""

  def variable_summaries(self, var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries' + var.name.replace(":", "")):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      # tf.summary.histogram('histogram', var)


  def do_summaries(self, graph, output_file="results/"):
    tf.summary.scalar("loss", self.agents.listener_loss)
    tf.summary.scalar("reward", tf.reduce_mean(self.agents.reward))
    # tf.summary.scalar("gradnorm", self.gradnorm)
    for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
      self.variable_summaries(var)
    merged = tf.summary.merge_all()
    return merged, tf.summary.FileWriter(output_file, graph)

  def __init__(self, concept_to_cat = None, batch_size = 32):
    self.batch_size = batch_size
    self.concept_to_cat = concept_to_cat

    self.idx_to_path = pickle.load(open("idx_to_path", 'r'))
    # self.optimizer = tf.train.AdamOptimizer(learning_rate)
    # gvs = self.optimizer.compute_gradients(self.agents.loss)
    #
    # gs, vs = zip(*gvs)
    # # if grad_clip:
    #   # clipped_gv = []
    #   # for grad, var in gvs:
    #   #   clipped_gv.append((tf.clip_by_norm(grad, 1), var))
    #   # gvs = clipped_gv
    #   # gs, _ = tf.clip_by_global_norm(gs, 5.0)
    # self.gradnorm = tf.global_norm(gs)
    #
    #
    # self.step = self.optimizer.apply_gradients(zip(gs, vs))



  class SpeakerValueEstimator():
    def __init__(self):

      self.in_size = 4096
      self.x_target = tf.placeholder(tf.float32, (None, self.in_size))
      self.x_distractor = tf.placeholder(tf.float32, (None, self.in_size))
      self.reward = tf.placeholder(tf.float32, (None))
      state = tf.concat([self.x_target, self.x_distractor], axis=1)
      with tf.variable_scope("value_est") as scope:
        self.in_layer = tf.contrib.layers.fully_connected(state, num_outputs=50, scope=scope)

        self.v_output_layer = tf.contrib.layers.fully_connected(self.in_layer, num_outputs=1, activation_fn=None, reuse=False)
      self.value_estimate = tf.squeeze(self.v_output_layer)

      self.value_loss = tf.reduce_mean(tf.squared_difference(self.value_estimate, self.reward))

      self.optimizer = tf.train.AdamOptimizer()
      self.train_op = self.optimizer.minimize(self.value_loss)

  # class ListenerValueEstimator():
  #   def __init__(self, in_size = 4096, vocab_size = 100):
  #     self.x_1 = tf.placeholder(tf.float32, (None, in_size))
  #     self.x_2 = tf.placeholder(tf.float32, (None, in_size))
  #     self.x_symbol = tf.placeholder(tf.float32, (None))
  #     self.reward = tf.placeholder(tf.float32, (None))
  #     state = tf.concat([self.x_1, self.x_2, ], axis=1)
  #     with tf.variable_scope("l_value_est") as scope:
  #       self.in_layer = tf.contrib.layers.fully_connected(state, num_outputs = 50, scope=scope)
  #       self.v_output_layer = tf.contrib.layers.fully_connected(self.in_layer, num_outputs=1, activation_fn=None, reuse=False)
  #       self.value_estimate = tf.squeeze(self.v_output_layer)
  #
  #       self.value_loss = tf.reduce_mean(tf.squared_difference(self.value_estimate, self.reward))
  #
  #       self.optimizer = tf.train.AdamOptimizer()
  #       self.train_op = self.optimizer.minimize(self.value_loss)

  def run3(self, sess, concepts_to_matrices, num_iters = 50000):
    self.agents = agents3.SpeakerListenerN(self.batch_size)
    # sve = self.SpeakerValueEstimator()
    # lve = self.ListenerValueEstimator()
    sess.run(tf.global_variables_initializer())
    name = "results" + datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    os.makedirs(name)
    merged, fw = self.do_summaries(sess.graph, output_file=name)
    # train, train_permute, _ = self.get_batch(50000, concepts_to_matrices)

    test, test_permute, test_record = self.get_batchN(10000, concepts_to_matrices)
    permuted_test = np.transpose(test[np.arange(10000), :, test_permute], axes=[1, 2, 0])
    vocab_size = 100
    symbols_total = np.zeros((num_iters / self.batch_size, vocab_size))

    for i in range(num_iters / self.batch_size):

      batch, permute, _ = self.get_batchN(self.batch_size, concepts_to_matrices)
      permuted_batch = np.transpose(batch[np.arange(self.batch_size), :, permute], axes=[1, 2, 0])

      # run agent prediction, get reward
      _, l_loss, s_loss, summary, indices, act, reward, out, scores, symbols, speaker_scores, resp_outputs = sess.run(
        [self.agents.train_op, self.agents.listener_loss, self.agents.speaker_loss, merged, self.agents.indexes, self.agents.train_act, self.agents.reward, self.agents.out, self.agents.scores,
         self.agents.symbol_idx, self.agents.speaker_scores, self.agents.responsible_outputs], feed_dict={
          self.agents.x_target: batch[:, :, 0],
          self.agents.x_distractor: batch[:, :, 1],
          self.agents.x_distractor2: batch[:, :, 2],
          self.agents.x_1: permuted_batch[:, :, 0],
          self.agents.x_2: permuted_batch[:, :, 1],
          self.agents.x_3:permuted_batch[:,:,2],
          self.agents.y: permute[0]
        })

      # # value estimate
      # value_estimate, _, v_loss = sess.run([sve.value_estimate, sve.train_op, sve.value_loss], feed_dict={
      #   sve.x_target: batch[:, :, 0],
      #   sve.x_distractor: batch[:, :, 1],
      #   sve.reward: reward
      # })
      # # update
      # _, l_loss, s_loss = sess.run([self.agents.train_op, self.agents.listener_loss, self.agents.speaker_loss],
      #                              feed_dict={
      #                                self.agents.x_target: batch[:, :, 0],
      #                                self.agents.x_distractor: batch[:, :, 1],
      #                                self.agents.x_1: permuted_batch[:, :, 0],
      #                                self.agents.x_2: permuted_batch[:, :, 1],
      #                                self.agents.y: permute[0],
      #                                self.agents.value_estimate: value_estimate
      #                              })

      # feed_dict = dict(zip(self.agents.gradient_holders, grad))

      fw.add_summary(summary, i)
      # syms = [s for j in symbols.tolist() for s in j]

      np.add.at(symbols_total[i], symbols, 1)

      # reward = np.mean(np.equals(acts, permute[:, :, 0]).astype(int))

      if i % 10 == 0:
        print(i, l_loss, s_loss, np.mean(reward))

    test_result, symbols_test = sess.run([self.agents.reward, self.agents.symbol_idx], feed_dict={
      self.agents.x_target: test[:, :, 0],
      self.agents.x_distractor: test[:, :, 1],
      self.agents.x_distractor2: test[:, :, 2],
      self.agents.x_1: permuted_test[:, :, 0],
      self.agents.x_2: permuted_test[:, :, 1],
      self.agents.x_3:permuted_test[:, :, 2],
      self.agents.y: test_permute[0]
      # self.agents.temp : 10
    })
    print(np.mean(test_result))
    # self.get_mistakes(test_result, test_record)
    # self.get_purity(test_record, symbols_test)
    # self.get_random_purity(test_record, symbols_test)

    symbol_counter = Counter()
    for i in range(len(symbols_test)):
      symbol_counter[symbols_test[i]] += 1

    print symbol_counter

    # compute moving average:
    averages = []
    totals = np.zeros(vocab_size)
    for i in range(vocab_size):
      averages.append([])
    for i in range(num_iters / self.batch_size):
      for j in range(vocab_size):
        totals[j] += symbols_total[i, j]
        averages[j].append(float(totals[j]) / (i + 1))

    for j in range(vocab_size):
      plt.plot(averages[j])
    print(len(symbol_counter.keys()))
    plt.show()
    print("")


  def run(self, sess, concepts_to_matrices, num_iters = 50000):
    self.agents = agents.SpeakerListener(self.batch_size)
    sve = self.SpeakerValueEstimator()
    # lve = self.ListenerValueEstimator()
    sess.run(tf.global_variables_initializer())
    name = "results" + datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    os.makedirs(name)
    merged, fw = self.do_summaries(sess.graph, output_file=name)
    # train, train_permute, _ = self.get_batch(50000, concepts_to_matrices)

    test, test_permute, test_record = self.get_batch(10000, concepts_to_matrices)
    permuted_test = np.transpose(test[np.arange(10000), :, test_permute], axes=[1, 2, 0])
    vocab_size = 10
    symbols_total = np.zeros((num_iters / self.batch_size, vocab_size))

    for i in range(num_iters / self.batch_size):

      batch, permute, _ =  self.get_batch(self.batch_size, concepts_to_matrices)
      permuted_batch = np.transpose(batch[np.arange(self.batch_size), :, permute], axes=[1, 2, 0])

      #run agent prediction, get reward
      summary, indices, act,reward,  out, scores, symbols, speaker_scores, resp_outputs = sess.run([ merged, self.agents.indexes, self.agents.train_act,  self.agents.reward, self.agents.out, self.agents.scores, self.agents.symbol_idx, self.agents.speaker_scores, self.agents.responsible_outputs], feed_dict={
        self.agents.x_target : batch[:, :, 0],
        self.agents.x_distractor : batch[:, :, 1],
        self.agents.x_1 : permuted_batch[:, :, 0],
        self.agents.x_2 : permuted_batch[:, :, 1],
        self.agents.y : permute[0]
      })

      #value estimate
      value_estimate, _, v_loss = sess.run([sve.value_estimate, sve.train_op, sve.value_loss], feed_dict = {
        sve.x_target : batch[:, :, 0],
        sve.x_distractor : batch[:, :, 1],
        sve.reward : reward
      })
      #update
      _ , l_loss, s_loss= sess.run([self.agents.train_op, self.agents.listener_loss, self.agents.speaker_loss], feed_dict = {
        self.agents.x_target: batch[:, :, 0],
        self.agents.x_distractor: batch[:, :, 1],
        self.agents.x_1: permuted_batch[:, :, 0],
        self.agents.x_2: permuted_batch[:, :, 1],
        self.agents.y: permute[0],
        self.agents.value_estimate : value_estimate
      })



      # feed_dict = dict(zip(self.agents.gradient_holders, grad))

      fw.add_summary(summary, i)
      # syms = [s for j in symbols.tolist() for s in j]

      np.add.at(symbols_total[i], symbols, 1)

      # reward = np.mean(np.equals(acts, permute[:, :, 0]).astype(int))

      if i % 10 == 0:
        print(i, l_loss, s_loss, np.mean(reward))


    test_result, symbols_test = sess.run([self.agents.reward, self.agents.symbol_idx], feed_dict = {
      self.agents.x_target : test[:, :, 0],
      self.agents.x_distractor : test[:, :, 1],
      self.agents.x_1 : permuted_test[:, :, 0],
      self.agents.x_2 : permuted_test[:, :, 1],
      self.agents.y : test_permute[0]
      # self.agents.temp : 10
    })
    print(np.mean(test_result))
    self.get_mistakes(test_result, test_record)
    self.get_purity(test_record, symbols_test)
    self.get_random_purity(test_record, symbols_test)

    symbol_counter = Counter()
    for i in range(len(symbols_test)):
        symbol_counter[symbols_test[i]] += 1

    print symbol_counter

    #compute moving average:
    averages = []
    totals = np.zeros(vocab_size)
    for i in range(vocab_size):
      averages.append([])
    for i in range(num_iters / self.batch_size):
      for j in range(vocab_size):
        totals[j] += symbols_total[i, j]
        averages[j].append(float(totals[j])/(i+1) / self.batch_size * 100)


    for j in range(vocab_size):
      plt.plot(averages[j])
    print(len(symbol_counter.keys()))
    plt.xlabel("Epochs")
    plt.ylabel("Symbol usage (%)")
    plt.title("Symbol average usage over time")
    plt.show()
    print("")











