"""
本模块用于统计加噪情况
"""
import numpy as np
from my_AttriGuard.input_data import Data



def compute_noise(target_label_list):
    """
    统计加噪情况：
    1. 各个样本在各类规避攻击下改变的属性值的数量；
    2. 当前该类的规避攻击下平均每个样本改变的属性值的数量；
    :param target_label_list: 规避攻击的target_label组成的列表
    :return: None
    """
    data = Data()

    noise_num = np.zeros([data.num_of_test, len(target_label_list)])  # 统计各个样本在各类规避攻击下改变的属性值的数量
    for i in target_label_list:
        filepath = "./data/adversarial_sample/" + str(i) + ".txt"
        adversarial_sample = np.loadtxt(filepath)
        overall_noise = 0.0  # 当前该类的规避攻击下，所有样本改变的属性值数量的总和
        for j in np.arange(data.num_of_test):
            difference = data.test_features[j, :] - adversarial_sample[j, :]
            nonzero = np.nonzero(difference)[0]
            noise_number = nonzero.shape[0]  # 第j个样本在第i类规避攻击下改变的属性值的数量
            noise_num[j, i] = noise_number
            overall_noise += noise_number
        print("niose added: {}".format(overall_noise / data.num_of_test))  # 当前该类的规避攻击下平均每个样本改变的属性值的数量
    np.savetxt("./data/adversarial_sample/noise.txt", noise_num)
