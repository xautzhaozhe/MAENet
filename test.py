from numpy.random import seed

seed(1)
from tensorflow import set_random_seed

set_random_seed(2)
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.io as sio
from sklearn import metrics
from scipy.spatial.distance import pdist, squareform


def ROC_AUC(target2d, groundtruth):
    """
    :param target2d: the 2D anomaly component
    :param groundtruth: the groundtruth
    :return: auc: the AUC value
    """
    rows, cols = groundtruth.shape
    label = groundtruth.transpose().reshape(1, rows * cols)
    target2d = target2d.transpose().reshape(1, rows * cols)
    result = np.zeros((1, rows * cols))
    for i in range(rows * cols):
        result[0, i] = np.linalg.norm(target2d[:, i])

    # result = hyper.hypernorm(result, "minmax")
    fpr, tpr, thresholds = metrics.roc_curve(label.transpose(), result.transpose())
    auc = metrics.auc(fpr, tpr)
    print('AUC value is: ', auc)

    return auc


def pixdensity(data, win_size, q):
    row, col, band = data.shape
    win_half = int(win_size / 2)
    pix_density = np.zeros((row, col))
    cutdis = win_size ** 2 * (win_size ** 2 - 1) * q / 100
    datatest = np.zeros((3 * row, 3 * col, band))
    # 对数据进行镜像填充
    datatest[row:2 * row, col:2 * col, :] = data
    datatest[row:2 * row, 0:col, :] = data[:, ::-1, :]
    datatest[row:2 * row, 2 * col:3 * col, :] = data[:, ::-1, :]
    center_data = datatest[row:2 * row, :, :]
    datatest[0:row, :, :] = center_data[::-1, :, :]
    datatest[2 * row:3 * row, :, :] = center_data[::-1, :, :]
    # 计算局部区域各点间的距离
    for i in np.arange(row, 2 * row):
        for j in np.arange(col, 2 * col):
            # 获取以每一个像素为中心的单窗口像素
            matrix = datatest[i - win_half:i + win_half + 1, j - win_half:j + win_half + 1, :]
            matrix = np.reshape(matrix, (win_size ** 2, band))
            # 计算中心像素与其余像素的距离
            dis_matrix = np.zeros(win_size ** 2)
            for e in range(win_size ** 2):
                dis_matrix[e] = (np.sum(np.square(matrix[int(win_size ** 2 / 2) + 1, :] - matrix[e, :])))**0.5
            # distList = pdist(matrix, metric='euclidean')
            # distMatrix = squareform(distList)

            # 去除掉大于截断距离的样本
            dis_gao = np.zeros(win_size ** 2)
            for n in range(win_size ** 2):
                if dis_matrix[n] - cutdis > 0:
                    continue
                # 用高斯函数计算局部密度
                dis_gao[n] = np.exp(-np.square(dis_matrix[n] / cutdis))
            dis_gao = np.mean(dis_gao)
            pix_density[i - row, j - col] = dis_gao
    pix_density = np.exp(-pix_density)

    return pix_density


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
    # 计算每个像素和均值向量之间的马氏距离 RX算法
    row, col, band = data.shape
    data = data.reshape(row * col, band)
    # 先求协方差矩阵
    mean_vector = np.mean(data, axis=0)
    mean_matrix = np.tile(mean_vector, (row * col, 1))
    re_matrix = data - mean_matrix
    matrix = np.dot(re_matrix.T, re_matrix) / (row * col - 1)
    # 在计算过程中有的矩阵是奇异阵，所有这里求的是伪逆矩阵
    variance_covariance = np.linalg.pinv(matrix)

    # 计算每个像素的马氏距离
    distances = np.zeros([row * col, 1])
    for i in range(row * col):
        re_array = re_matrix[i]
        re_var = np.dot(re_array, variance_covariance)
        distances[i] = np.dot(re_var, np.transpose(re_array))
    distances = distances.reshape(row, col)

    return distances


if __name__ == "__main__":

    data_name = 'GrandIsle'

    dataset = sio.loadmat('./data/{}.mat'.format(data_name))['data']
    dataset = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))
    gt = sio.loadmat('./data/{}.mat'.format(data_name))['map']
    re_data = sio.loadmat('./test/{}-re.mat'.format(data_name))['data']
    enc_data = sio.loadmat('./test/{}-enc.mat'.format(data_name))['data']
    # print(re_data)

    desity_encdata = pixdensity(enc_data, win_size=15, q=2)

    plt.imshow(desity_encdata)
    plt.title('density of encdata:')
    plt.show()

    print('orginal dataset shape: ', dataset.shape)
    img = dataset[:, :, 25]
    plt.imshow(img)
    plt.show()
    ma = Mahalanobis(re_data)
    residual1 = residual(dataset, re_data)
    sio.savemat('./results/ourAE_{}.mat'.format(data_name), {'our': residual1})
    plt.imshow(residual1)
    plt.show()

    print('residual image: ')
    ROC_AUC(residual1, gt)
    print('RX on reconstruction image: ')
    ROC_AUC(ma, gt)
    print('RX and residual error: ')
    mix_img = residual1 + ma
    ROC_AUC(mix_img, gt)
    print('density of encdata result:')
    ROC_AUC(desity_encdata, gt)

    print('density plus residual:')
    mix_encdensity = 0.5*desity_encdata + 0.5*residual1
    sio.savemat('./results/our_{}.mat'.format(data_name), {'result':mix_encdensity})
    ROC_AUC(mix_encdensity, gt)
