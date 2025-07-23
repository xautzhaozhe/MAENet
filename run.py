# coding: utf-8
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
import os, warnings
import tensorflow as tf
import datamanager as dman
import neuralnet as nw
import tf_process as tfp
import scipy.io as sio
import matplotlib.pylab as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
warnings.filterwarnings('ignore')
tf.disable_eager_execution()


if __name__ == '__main__':

    # alpha and batch are import for MAENet
    # bt=10000  for urban-2, urban-4;  BT=75000 for Segundo;  Bt=64 for AVIRIS-2, AVIRIS-3;

    data_name = 'abu-urban-2'

    lr = 1e-4
    epoch = 50
    batch = 10000
    n = 320
    alpha = 0.01

    dataset = sio.loadmat('./data/{}.mat'.format(data_name))['data']
    print('original dataset shape: ', dataset.shape)
    row, col, channel = dataset.shape
    sample = row*col
    dataset = dman.Dataset(normalize=True, img=dataset)
    neuralnet = nw.MemAE(dataset, channel=channel, alpha=alpha, leaning_rate=lr, n=n)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    tfp.training(sess=sess, neuralnet=neuralnet, saver=saver, dataset=dataset, epochs=epoch,
                 batch_size=batch)

    tfp.test(sess=sess, neuralnet=neuralnet, saver=saver, dataset=dataset, row=row, col=col,
             channel=channel, batch_size=sample, data_name=data_name, enc_dim=neuralnet.enc_d)







