# coding:utf-8
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
import tensorflow as tf


class MemAE(object):

    def __init__(self, dataset, channel, alpha, n, leaning_rate=1e-3):

        print("\nInitializing Neural Network...")
        self.dataset = dataset
        self.n = n
        self.alpha, self.leaning_rate, = alpha, leaning_rate
        self.training = False

        self.x = tf.placeholder(tf.float32, [None, channel])
        self.batch_size = tf.placeholder(tf.int32, shape=None)

        self.weights, self.biasis = [], []
        self.w_names, self.b_names = [], []
        self.fc_shapes, self.conv_shapes = [], []

        self.x_hat, self.w_hat = self.build_model(input=self.x, channel=channel, n=self.n)
        self.z_enc, self.enc_d = self.encoder(input=self.x)
        self.z_hat, self.addre = self.memory(input=self.z_enc, n=n, c=self.enc_d)

        self.mse_r = self.mean_square_error(x1=self.x, x2=self.x_hat)
        self.mem_etrp = tf.reduce_sum((-self.w_hat) * tf.math.log(self.w_hat + 1e-12))
        self.loss = tf.reduce_mean(self.mse_r + (self.alpha * self.mem_etrp))
        self.optimizer = tf.train.AdamOptimizer(
            self.leaning_rate, beta1=0.9, beta2=0.999).minimize(self.loss)

        tf.summary.scalar('MemAE/mse', tf.reduce_sum(self.mse_r))
        tf.summary.scalar('MemAE/w-entropy', tf.reduce_sum(self.mem_etrp))
        tf.summary.scalar('MemAE/total loss', self.loss)
        self.summaries = tf.summary.merge_all()

    def set_training(self):
        self.training = True

    def set_test(self):
        self.training = False

    def mean_square_error(self, x1, x2):

        data_dim = len(x1.shape)
        if data_dim == 4:
            return tf.reduce_sum(tf.square(x1 - x2), axis=(1, 2, 3))
        elif data_dim == 3:
            return tf.reduce_sum(tf.square(x1 - x2), axis=(1, 2))
        elif data_dim == 2:
            return tf.reduce_sum(tf.square(x1 - x2), axis=1)
        else:
            return tf.reduce_sum(tf.square(x1 - x2))

    def cosine_sim(self, x1, x2):
        num = tf.matmul(x1, tf.transpose(x2), name='attention_num')
        denom = tf.matmul(x1 ** 2, tf.transpose(x2) ** 2, name='attention_denum')
        w = (num + 1e-12) / (denom + 1e-12)

        return w

    def build_model(self, input, channel, n):

        with tf.name_scope('encoder') as scope_enc:
            z_enc, z_c = self.encoder(input=input)
        with tf.name_scope('memory') as scope_enc:
            z_hat, w_hat = self.memory(input=z_enc, n=n, c=z_c)

        with tf.name_scope('decoder') as scope_enc:
            x_hat = self.decoder(input=z_hat, channel=channel)

        return x_hat, w_hat

    def encoder(self, input):

        print("Encode")

        lay1 = tf.layers.dense(inputs=input, units=128)
        bn1 = self.batch_normalization(input=lay1, name="bn1")
        act1 = tf.nn.leaky_relu(bn1, alpha=0.2)

        lay2 = tf.layers.dense(inputs=act1, units=64)
        bn2 = self.batch_normalization(input=lay2, name="bn2")
        act2 = tf.nn.leaky_relu(bn2, alpha=0.2)

        lay3 = tf.layers.dense(inputs=act2, units=18)
        act3 = tf.nn.sigmoid(lay3)

        [n, c] = act3.shape
        z = act3

        return z, c

    def memory(self, input, n, c=24):

        # N = Memory Capacity
        self.weights, self.w_names, w_memory = self.variable_maker(var_bank=self.weights, name_bank=self.w_names,
                                                                   shape=[n, c], name='w_memory')

        print("Attention for Memory Addressing")

        cosim = self.cosine_sim(x1=input, x2=w_memory)  # Eq.5
        atteniton = tf.nn.softmax(cosim)  # Eq.4

        print("input shape: ", input.shape, '*****', "memory shape: ", w_memory.shape, '*****', 'Attention shape: ',
              atteniton.shape)

        print("Hard Shrinkage for Sparse Addressing")

        lam = 2 / n  # deactivate the 1/N of N memories.

        addr_num = tf.compat.v1.nn.relu(atteniton - lam) * atteniton
        addr_denum = tf.abs(atteniton - lam) + 1e-12  # Eq 6
        memory_addr = addr_num / addr_denum
        renorm = memory_addr
        z_hat = tf.compat.v1.matmul(renorm, w_memory, name='shrinkage')
        print("Shrinkage w：", renorm.shape, '*****', "Memory shape：", w_memory.shape, '*****', "z_hat shape：",
              z_hat.shape)
        print('z_hat:', z_hat)

        return z_hat, renorm

    def decoder(self, input, channel):

        print("Decode")
        layt1 = tf.layers.dense(inputs=input, units=64)
        bnt1 = self.batch_normalization(input=layt1, name="bnt1")
        actt1 = tf.nn.leaky_relu(bnt1, alpha=0.2)

        layt2 = tf.layers.dense(inputs=actt1, units=128)
        bnt2 = self.batch_normalization(input=layt2, name="bnt2")
        actt2 = tf.nn.leaky_relu(bnt2, alpha=0.2)

        layt3 = tf.layers.dense(inputs=actt2, units=channel)
        bnt3 = self.batch_normalization(input=layt3, name="bnt2")
        x_hat = tf.nn.leaky_relu(bnt3, alpha=0.2)

        return x_hat

    def initializer(self):
        return tf.initializers.variance_scaling(distribution="untruncated_normal", dtype=tf.dtypes.float32)

    def variable_maker(self, var_bank, name_bank, shape, name=""):

        try:
            var_idx = name_bank.index(name)
        except:
            variable = tf.get_variable(name=name,
                                                 shape=shape, initializer=self.initializer())

            var_bank.append(variable)
            name_bank.append(name)
        else:
            variable = var_bank[var_idx]

        return var_bank, name_bank, variable

    def batch_normalization(self, input, name=""):
        bnlayer = tf.keras.layers.BatchNormalization(
            axis=-1,
            momentum=0.99,
            epsilon=0.001,
            center=True,
            scale=True,
            beta_initializer='zeros',
            gamma_initializer='ones',
            moving_mean_initializer='zeros',
            moving_variance_initializer='ones',
            renorm_momentum=0.99,
            trainable=True,
            name="%s_bn" % name,
        )

        bn = bnlayer(inputs=input, training=self.training)
        return bn

