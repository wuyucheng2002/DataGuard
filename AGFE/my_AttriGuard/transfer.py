"""
本模块用于保存attack model对于加噪后数据的预测结果
"""
import tensorflow as tf
import numpy as np
from my_AttriGuard.model import nn_model_attack
from my_AttriGuard.input_data import Data


def transfer(learning_rate, target_label_list):
    """
    保存attack model对于各个加噪后数据的预测结果
    :param learning_rate: nn_attack的学习率
    :param target_label_list: 规避攻击的target_label组成的列表
    :return: None
    """
    data = Data()

    # 加载训练好的nn_attack
    npzdata = np.load("model/nn_model_attack.npz", allow_pickle=True)
    weights = npzdata['w']
    model = nn_model_attack(data.num_of_features, data.num_of_label)
    for i in range(len(model.layers)):
        model.layers[i].trainable = False
    model.compile(loss=tf.keras.losses.binary_crossentropy,
                  optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate))
    model.build(input_shape=(None, data.num_of_features))
    model.set_weights(weights)
    model.summary()

    # 保存attacker分类器对加噪后数据预测的结果，每一行是一个test_user，每一列是规避攻击为i时，attacker分类器的预测结果
    attack_prediction_value = np.zeros([data.num_of_test, data.num_of_label])
    for i in target_label_list:
        filepath = "./data/adversarial_sample/" + str(i) + ".txt"
        adversarial_sample = np.loadtxt(filepath)
        pred = model(adversarial_sample)
        pred_value = np.argmax(pred, axis=1)
        attack_prediction_value[:, i] = pred_value[:]
    np.savetxt('./data/attack_prediction/nn.txt', attack_prediction_value)
