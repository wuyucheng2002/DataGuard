"""
本模块用于训练各个component
"""
import tensorflow as tf
from my_TIPRDC.data_prepare import Data
from my_TIPRDC.component import info_loss, evaluate
from my_TIPRDC.component import Feature_Extractor, Classifier, MutlInfo

criterion = tf.keras.losses.BinaryCrossentropy()  # loss计算准则
# 联合训练3个component
global_loss = []  # 总的目标函数的loss
target_loss = []  # classifier的loss
information = []  # mutil information estimator估计的互信息
# 单独训练classifier
cf_train_loss = []  # classifier的train_loss
cf_test_loss = []  # classifier的test_loss
cf_test_acc = []  # classifier的test_acc


def train(FE, CF, MI, data, batch_size=256, lr=0.1, lbd=0.66):
    """
    联合训练feature extractor, classifier, mutil information estimator
    :param FE: feature extractor
    :param CF: classifier
    :param MI: mutil information estimator
    :param batch_size:
    :param lr:
    :param lbd: 用于控制效用与隐私的tradeoff, lbd越大，效用损失越多
    :return: (FE, CF, MI)
    """
    data_iter = data.load_array((data.train_features_extractor, data.train_label_extractor), batch_size=batch_size,
                                is_train=True)

    optimizer_FE = tf.keras.optimizers.SGD(learning_rate=lr)
    optimizer_CF = tf.keras.optimizers.SGD(learning_rate=lr)
    optimizer_MI = tf.keras.optimizers.SGD(learning_rate=lr)

    for x, y in data_iter:
        with tf.GradientTape(persistent=True) as tape:
            x_prime = tf.concat([x[1:], tf.expand_dims(x[0], 0)], axis=0)  # x_prime就是把x的第一个样本位置换到最后一个
            u = tf.one_hot(tf.cast(y, tf.int32), depth=data.num_of_label_u)
            print(u.shape)
            z = FE(x)
            output = CF(z)

            loss_target = criterion(tf.one_hot(tf.cast(y, tf.int32), depth=data.num_of_label_u), output)
            loss_jsd = -info_loss(MI, x, z, u, x_prime)
            loss = - lbd * loss_target + (1. - lbd) * loss_jsd

        grads_FE = tape.gradient(loss, FE.trainable_variables)
        grads_CF = tape.gradient(loss, CF.trainable_variables)
        grads_MI = tape.gradient(loss, MI.trainable_variables)

        optimizer_FE.apply_gradients(zip(grads_FE, FE.trainable_variables))
        optimizer_CF.apply_gradients(zip(grads_CF, CF.trainable_variables))
        optimizer_MI.apply_gradients(zip(grads_MI, MI.trainable_variables))

        global_loss.append(loss)
        target_loss.append(loss_target)
        information.append(loss_jsd)

    print("联合训练下总目标函数的loss:{}".format(loss.numpy()))

    return FE, CF, MI


def train_Z(FE, CF, data, batch_size=256, lr=0.1):
    """
    仅对抗训练feature extractor, classifier
    对于feature extractor旨在降低classifier(attacker)的acc
    对于classifier旨在提升自身的acc
    :param FE: feature extractor
    :param CF: classifier
    :param lr:
    :param batch_size:
    :return: (FE, CF)
    """
    data_iter = data.load_array((data.train_features_extractor, data.train_label_extractor), batch_size=batch_size,
                                is_train=True)

    optimizer_FE = tf.keras.optimizers.SGD(learning_rate=lr)
    optimizer_CF = tf.keras.optimizers.SGD(learning_rate=lr)

    for x, y in data_iter:
        with tf.GradientTape(persistent=True) as tape:
            z = FE(x)
            u = CF(z)
            loss_target = criterion(u, tf.one_hot(tf.cast(y, tf.int32), depth=data.num_of_label_u))
            loss = - loss_target

        grads_FE = tape.gradient(loss, FE.trainable_variables)
        grads_CF = tape.gradient(loss_target, CF.trainable_variables)

        optimizer_FE.apply_gradients(zip(grads_FE, FE.trainable_variables))
        optimizer_CF.apply_gradients(zip(grads_CF, CF.trainable_variables))

        global_loss.append(loss)

    print("对抗训练下，defense目标函数的loss:{}".format(loss.numpy()))

    return FE, CF


def train_classifier(FE, CF, data, batch_size=256, lr=0.1):
    """
    仅训练classifier，旨在提升自身的acc
    :param FE: feature extractor
    :param CF: classifier
    :param lr:
    :param batch_size:
    :return: CF
    """
    data_iter = data.load_array((data.train_features_extractor, data.train_label_extractor), batch_size=batch_size,
                                is_train=True)

    optimizer_CF = tf.keras.optimizers.SGD(learning_rate=lr)

    for x, y in data_iter:
        with tf.GradientTape() as tape:
            features = FE(x)
            output = CF(features)
            loss = criterion(output, tf.one_hot(tf.cast(y, tf.int32), depth=data.num_of_label_u))
        grads = tape.gradient(loss, CF.trainable_variables)
        optimizer_CF.apply_gradients(zip(grads, CF.trainable_variables))

        cf_train_loss.append(loss)

    return CF


def test_classifier(FE, CF, data):
    """
    查看classifier在测试集上的表现，即查看隐私保护得如何，表现得越差，隐私保护效果越好
    :param FE: feature extractor
    :param CF: classifier
    :return: None
    """
    features = FE(data.test_features_extractor)
    output = CF(features)
    loss = criterion(output, tf.one_hot(data.test_label_extractor, depth=data.num_of_label_u))
    acc = evaluate(y_true=data.test_label_extractor, y_pred=output)

    print('classifier | test_loss: %f, test_acc: %f' % (loss, acc))

    return acc

    # cf_test_loss.append(loss)
    # cf_test_acc.append(acc)


if __name__ == "__main__":
    FE = Feature_Extractor()
    CF = Classifier()
    MI = MutlInfo()
    data = Data()
    # train(FE, CF, MI, data)
    # train_Z(FE, CF, data)
    # train_classifier(FE, CF, data)
    test_classifier(FE, CF, data)
