import tensorflow as tf
import tensorflow.contrib.layers as layers

#IDEAS: Saliency maps!!! Will need to use your VGG though.
# Multiple agents in a population, take turns.
# Consider putting two separate loss functions: one for each agent. (Feed rewards directly to speaker

class SpeakerListener:
  def __init__(self, batch_size, in_size=4096, temp=10.):
    self.batch_size = batch_size
    self.in_size = in_size
    self.temp = temp

    self.init_speaker()

    self.init_listener()

#agnostic: 0.01

  def init_speaker(self, informed = False, hidden_size=50, num_filters = 20, vocab_size=10, is_train = True):

    self.x_target = tf.placeholder(tf.float32, (None, self.in_size))
    self.x_distractor = tf.placeholder(tf.float32, (None, self.in_size))

    with tf.variable_scope("speaker") as scope:


    #informed
      if informed:
        h_target = layers.fully_connected(self.x_target, hidden_size, activation_fn=None, scope=scope)
        h_distractor = layers.fully_connected(self.x_distractor, hidden_size, activation_fn=None, scope=scope,
                                              reuse=True)
        x = tf.expand_dims(tf.stack([h_target, h_distractor], axis=2), -1) # (batch, hidden, 2)
        x_conv = tf.layers.conv2d(x, num_filters, [1,2], activation=tf.sigmoid, use_bias=False) # (batch, hidden, 20)
        x_conv_sw = tf.transpose(x_conv, perm=[0,2,3,1])
        x_out = tf.layers.conv2d(x_conv_sw, vocab_size, [1, num_filters], activation=None, use_bias=False) #input filters kernel size
      else:
        h_target = layers.fully_connected(self.x_target, hidden_size, activation_fn=tf.sigmoid, scope=scope)
        h_distractor = layers.fully_connected(self.x_distractor, hidden_size, activation_fn=tf.sigmoid, scope=scope,
                                              reuse=True)
        h_combined = tf.concat([h_target, h_distractor], axis=1)
        x_out = layers.fully_connected(h_combined, vocab_size, activation_fn=None)


      # "Boltzmann"-
      self.speaker_out = tf.reshape(x_out / self.temp, [-1, vocab_size])
      self.speaker_scores = tf.reshape(tf.nn.softmax(x_out / self.temp), [-1, vocab_size])

      self.symbol_idx = tf.reshape(tf.multinomial(self.speaker_out, 1), [-1])
      self.symbol = tf.reshape(tf.one_hot(self.symbol_idx, vocab_size), [-1, vocab_size])

      self.speaker_indexes = tf.range(0, tf.shape(self.speaker_scores)[0]) * tf.shape(self.speaker_scores)[1] + tf.cast(self.symbol_idx, tf.int32)
      self.speaker_responsible_outputs = tf.gather(tf.reshape(self.speaker_scores, [-1]), self.speaker_indexes)

      # self.s_indexes = tf.range(0, tf.shape(self.speaker_scores)[0]) * tf.shape(self.speaker_scores)[1] + self.symbol_idx
      # self.s_responsible_outputs = tf.gather(tf.reshape(self.speaker_scores, [-1]), self.s_indexes)




  def supervisedSetting(self):
    pass



  def init_listener(self, hidden_size = 50, is_train = True):

    self.x_1 = tf.placeholder(tf.float32, (None, self.in_size))
    self.x_2 = tf.placeholder(tf.float32, (None, self.in_size))

    with tf.variable_scope("listener") as scope:
      h_1 = layers.fully_connected(self.x_1, hidden_size, activation_fn=None, scope=scope)
      h_2 = layers.fully_connected(self.x_2, hidden_size, activation_fn = None, scope=scope, reuse=True)

      sym_embed = tf.transpose(layers.fully_connected(self.symbol, hidden_size))

    # else:
    #   sym_embed = tf.transpose(layers.fully_connected(self.symbol, hidden_size))
    dot1 = tf.diag_part(tf.matmul(h_1, sym_embed))  #batch_size, 1
    dot2 = tf.diag_part(tf.matmul(h_2, sym_embed)) #batch_size, 1

    self.out = tf.stack([dot1, dot2], axis=1) / self.temp  #batch_size, 2
    self.scores = tf.nn.softmax(self.out)
    # self.value_estimator =
    self.y = tf.placeholder(tf.int32, (None))

    self.train_act = self.train_act() #sample
    # self.test_act = self.test_act() #argmax

    ### training:

    self.reward = tf.cast(tf.equal(self.train_act, self.y), tf.float32)
    self.reward.set_shape([self.batch_size])
    self.ave_reward = tf.reduce_mean(self.reward)

    self.value_estimate = tf.placeholder(tf.float32, (None))

    #update model

    self.baseline_est = tf.train.ExponentialMovingAverage(decay = 0.999)
    self.indexes = tf.range(0, tf.shape(self.scores)[0]) * tf.shape(self.scores)[1] + self.train_act

    self.responsible_outputs = tf.gather(tf.reshape(self.scores, [-1]), self.indexes)
    maintain_averages_op = self.baseline_est.apply([self.reward])
    self.baseline_ave = self.baseline_est.average(self.reward)
    # self.listener_loss = tf.reduce_mean(tf.log(tf.clip_by_value(self.responsible_outputs, 1e-10, 1.0))*(self.reward)) #
    # self.speaker_loss = tf.reduce_mean(tf.log(tf.clip_by_value(self.speaker_responsible_outputs, 1e-10, 1.0))* (self.reward))
    mean, var = tf.nn.moments(self.reward, axes=[0])
    self.std_reward = (self.reward)
    self.l_ll = tf.log(tf.clip_by_value(self.responsible_outputs, 1e-10, 10.0))
    self.s_ll = tf.log(tf.clip_by_value(self.speaker_responsible_outputs, 1e-10, 10.0))
    self.optimizer = tf.train.AdamOptimizer(learning_rate=0.005)
    # self.SpeakerValueEstimator()

    self.listener_loss = -tf.reduce_mean(self.l_ll * (self.std_reward - self.baseline_ave)) #
    self.speaker_loss = -tf.reduce_mean(self.s_ll * (self.std_reward - self.baseline_ave))
    # self.listener_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.y, logits = self.out))
    # self.speaker_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.symbol_idx, logits = self.speaker_out))
    # self.tr_listener_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.y, logits = self.tr_out))

    #gradients
    # tvars = tf.trainable_variables()

    # self.speaker_gradients = optimizer.compute_gradients(self.speaker_loss)
    self.listener_gradients = self.optimizer.compute_gradients(self.listener_loss)
    self.speaker_gradients = self.optimizer.compute_gradients(self.speaker_loss)

    # for i in range(len(self.gradients)):
    #   self.gradients[i] = tf.clip_by_norm(self.gradients[i], 5)

    # self.gradient_holders = []
    # for idx, var in enumerate(tvars):
    #   placeholder = tf.placeholder(tf.float32,name=str(idx)+'_holder')
    #   self.gradient_holders.append(placeholder)


    # self.update_batch = optimizer.apply_gradients(self.speaker_gradients)
    #
    # for i, (grad, var) in enumerate(self.listener_gradients):
    #   if grad is not None:
    #     self.listener_gradients[i] = (grad * self.ave_reward, var)

    self.gradients = []
    for (grad,var), (grad_s, var_s) in zip(self.listener_gradients, self.speaker_gradients):
      if grad is not None:
        self.gradients.append((grad,var))
      else:
        self.gradients.append((grad_s, var_s))
    # a = lambda :self.speaker_gradients

    # self.gradients = tf.cond(self.bool, lambda:self.speaker_gradients, lambda:self.listener_gradients)

    self.update_batch = self.optimizer.apply_gradients(self.gradients)

    # with tf.control_dependencies([self.update_batch]):
    with tf.control_dependencies([self.update_batch]):
      self.train_op = tf.group(maintain_averages_op)


  def train_act(self):
    return tf.reshape(tf.cast(tf.multinomial(self.out, 1), tf.int32), [-1]) # acts for each ex in batch

  # def loss(self):
  #   self.indices = tf.stack([tf.range(0, self.batch_size),  self.y], axis=1)
  #   self.test = tf.gather_nd(self.scores, self.indices)
  #   self.ave_reward = tf.reduce_mean(self.test)
  #
  #   return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.out))





