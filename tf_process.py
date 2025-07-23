# coding: utf-8

import os, inspect, math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn import metrics

PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + "/.."


def ROC_AUC(target2d, groundtruth):
    rows, cols = groundtruth.shape
    label = groundtruth.transpose().reshape(1, rows * cols)
    target2d = target2d.transpose().reshape(1, rows * cols)
    result = np.zeros((1, rows * cols))
    for i in range(rows * cols):
        result[0, i] = np.linalg.norm(target2d[:, i])
    fpr, tpr, thresholds = metrics.roc_curve(label.transpose(), result.transpose())
    auc = metrics.auc(fpr, tpr)
    print('AUC value is: ', auc)

    return auc


def residual(contr_data, org_data):
    row, col, band = org_data.shape
    residual = (org_data - contr_data) ** 2
    result = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            R = np.mean(residual[i, j, :])
            # result[i, j] = np.exp(R)
            result[i, j] = R
    result = (result - np.min(result)) / (np.max(result) - np.min(result))

    return result


def Mahalanobis(data):
    row, col, band = data.shape
    data = data.reshape(row * col, band)
    mean_vector = np.mean(data, axis=0)
    mean_matrix = np.tile(mean_vector, (row * col, 1))
    re_matrix = data - mean_matrix
    matrix = np.dot(re_matrix.T, re_matrix) / (row * col - 1)
    variance_covariance = np.linalg.pinv(matrix)

    distances = np.zeros([row * col, 1])
    for i in range(row * col):
        re_array = re_matrix[i]
        re_var = np.dot(re_array, variance_covariance)
        distances[i] = np.dot(re_var, np.transpose(re_array))
    distances = distances.reshape(row, col)

    return distances


def make_dir(path):
    try:
        os.mkdir(path)
    except:
        pass


def training(sess, saver, neuralnet, dataset, epochs, batch_size):
    print("\nTraining to %d epochs (%d of minibatch size)" % (epochs, batch_size))

    summary_writer = tf.summary.FileWriter(PACK_PATH + '/Checkpoint', sess.graph)
    make_dir(path="results")

    iteration = 0
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    for epoch in range(epochs):
        x_tr, y_tr, _ = dataset.next_train(batch_size=batch_size, fix=True)  # Initial batch

        z_hat = sess.run(neuralnet.z_hat,
                         feed_dict={neuralnet.x: x_tr, neuralnet.batch_size: x_tr.shape[0]})

        while True:
            x_tr, y_tr, terminator = dataset.next_train(batch_size)

            neuralnet.set_training()
            _, summaries = sess.run([neuralnet.optimizer, neuralnet.summaries],
                                    feed_dict={neuralnet.x: x_tr, neuralnet.batch_size: x_tr.shape[0]},
                                    options=run_options, run_metadata=run_metadata)

            mse, w_etrp, loss = sess.run([neuralnet.mse_r, neuralnet.mem_etrp, neuralnet.loss],
                                         feed_dict={neuralnet.x: x_tr, neuralnet.batch_size: x_tr.shape[0]})
            summary_writer.add_summary(summaries, iteration)

            iteration += 1
            if terminator:
                break

        print("Epoch [%d / %d] (%d iteration)  MSE: %.3f, W-ETRP: %.3f, Total: %.3f" \
              % (epoch, epochs, iteration, mse.sum(), w_etrp.sum(), loss))

        saver.save(sess, PACK_PATH + "/Checkpoint/model_checker", global_step=epoch)
        summary_writer.add_run_metadata(run_metadata, 'epoch-%d' % epoch)


def test(sess, saver, neuralnet, dataset, row, col, channel, batch_size, data_name, enc_dim):
    print("\nTest...")

    if os.path.exists(PACK_PATH + "/Checkpoint/model_checker.index"):
        print("\nRestoring parameters")
        model_file = tf.train.latest_checkpoint('ckpt/')
        saver.restore(sess, model_file)

    make_dir(path="test")

    while True:
        x_te, y_te, terminator = dataset.next_test(batch_size)  # y_te does not used in this prj.

        x_restore = sess.run(neuralnet.x_hat, feed_dict={neuralnet.x: x_te, neuralnet.batch_size: x_te.shape[0]})
        print('re_data shape: ', x_restore.shape)

        enc_data = sess.run(neuralnet.z_enc, feed_dict={neuralnet.x: x_te, neuralnet.batch_size: x_te.shape[0]})
        print('enc_data shape: ', enc_data.shape)

        mem_data = sess.run(neuralnet.z_hat, feed_dict={neuralnet.x: x_te, neuralnet.batch_size: x_te.shape[0]})
        print('mem_data shape: ', mem_data.shape)

        if terminator:
            break

    print('reconstructed data shape：', x_restore.shape)
    re_data = np.reshape(x_restore, (row, col, channel))
    sio.savemat('./test/{}-re.mat'.format(data_name), {'data': re_data})
    dataset = sio.loadmat('./data/{}.mat'.format(data_name))['data']
    dataset = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))
    gt = sio.loadmat('./data/{}.mat'.format(data_name))['map']

    residual1 = residual(dataset, re_data)
    sio.savemat('./results/our_{}.mat'.format(data_name), {'our': residual1})
    plt.imshow(residual1)
    plt.title('residual image')
    plt.show()
    print('residual image: ')
    ROC_AUC(residual1, gt)




