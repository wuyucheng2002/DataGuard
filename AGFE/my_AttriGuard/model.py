"""
本模块存放defense和attacker的分类模型
目前二者均使用神经网络作为分类模型
其中defense隐层的神经元数目是attacker的一半
"""
import tensorflow as tf
import numpy as np


def nn_model_defense(input_size, num_classes):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=input_size, kernel_initializer=tf.keras.initializers.Zeros()))
    model.add(tf.keras.layers.Dense(units=50, kernel_initializer=tf.keras.initializers.RandomNormal(seed=42),
                                    activation='softmax'))
    model.add(tf.keras.layers.Dense(num_classes))
    model.add(tf.keras.layers.Activation('sigmoid'))

    return model


def nn_model_attack(input_size, num_classes):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=input_size, kernel_initializer=tf.keras.initializers.Zeros()))
    model.add(tf.keras.layers.Dense(units=100, kernel_initializer=tf.keras.initializers.RandomNormal(seed=42),
                                    activation='softmax'))
    model.add(tf.keras.layers.Dense(num_classes))
    model.add(tf.keras.layers.Activation('sigmoid'))

    return model


def evaluate(y_true, y_pred):
    """
    评估函数：计算accuracy_score
    :param y_true: such as [[1, 0], [0, 1], [1, 0]]
    :param y_pred: such as [[0.8, 0.2], [0.1, 0.9], [0.4, 0.6]]
    :return: accuracy_score, such as 0.67 for above example
    """
    y_pred = tf.argmax(y_pred, axis=1)
    res = tf.equal(tf.cast(y_true, y_pred.dtype), y_pred)
    res = np.mean(res)
    return res
