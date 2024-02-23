"""
本模块存放各component：
1. feature_extractor model(defense)
2. classifier model(attacker)
3. multi information estimator
"""
import tensorflow as tf
import numpy as np


class Feature_Extractor(tf.keras.Model):
    """
    feature_extractor(defense)
    尽可能从raw data中提取features并使得attacker无法反推出敏感属性u的信息
    包含：
    1. Dense(embedding_dim)
    """

    def __init__(self, embedding_dim=100):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.embedding = tf.keras.layers.Dense(embedding_dim)

    def call(self, X):
        embedding = self.embedding(X)
        return embedding


class Classifier(tf.keras.Model):
    """
    classifier(attacker)
    尽可能从extracted features中反推出敏感属性u的信息
    包含：
    1. Dense(hidden_dim, activation='softmax')
    2. Dense(target_size, activation='sigmoid')
    """

    def __init__(self, target_size, hidden_dim=100):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(hidden_dim, activation='softmax')
        self.dense2 = tf.keras.layers.Dense(target_size, activation='sigmoid')

    def call(self, X):
        output_dense1 = self.dense1(X)
        output_dense2 = self.dense2(output_dense1)
        return output_dense2


class MutlInfo(tf.keras.Model):
    """
    multi information estimator，用于计算E(x; z, u)
    包含：
    1. Dense(embedding_dim)
    2. Dense(self.hidden_dim, activation='softmax')
    3. Dense(self.hidden_dim, activation='softmax')
    4. Dense(self.target_size, activation='sigmoid')
    """
    def __init__(self, embedding_dim=100, hidden_dim=100, target_size=2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.target_size = target_size

        self.embedding = tf.keras.layers.Dense(self.embedding_dim)
        self.dense1 = tf.keras.layers.Dense(self.hidden_dim, activation='softmax')
        self.dense2 = tf.keras.layers.Dense(self.hidden_dim, activation='softmax')

        self.dense3 = tf.keras.layers.Dense(self.target_size, activation='sigmoid')

    def call(self, x, z, u):
        x = self.embedding(x)
        x = self.dense1(x)
        x = self.dense2(x)

        z = self.dense2(z)

        out = tf.concat([x, z, u], axis=1)
        out = self.dense3(out)

        return out


def info_loss(MI, x, z, u, x_prime):
    """
    用于计算I_JSD(x;z,u),我们希望最大化I_JSD(x; z, u)，即最大化extracted feature与raw data之间的互信息
    :param MI: mutil information estimator
    :param x: raw data
    :param z: extracted embedding
    :param u: extractor task label
    :param x_prime: random input data sampled independently from the same distribution of x
    :return: I_JSD(x;z;u)=Ej-Em
    """
    Ej = tf.reduce_mean(tf.nn.softplus(-MI(x, z, u)))
    Em = tf.reduce_mean(tf.nn.softplus(MI(x_prime, z, u)))
    return Ej - Em


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
