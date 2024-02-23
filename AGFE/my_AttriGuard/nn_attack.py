"""
本模块用于训练并保存nn_attack
"""
from my_AttriGuard.model import nn_model_attack, evaluate
from my_AttriGuard.input_data import Data
import tensorflow as tf
import numpy as np


def train_nn_attack(batch_size, learning_rate, epochs):
    """
    训练并保存nn_defense
    """
    data = Data()

    data_iter = data.load_array((data.train_features, data.train_label), batch_size=batch_size,
                                is_train=True)  # 获取训练数据迭代器
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    model = nn_model_attack(data.num_of_features, data.num_of_label)

    for epoch in range(epochs):
        train_loss_list = []  # 存放每个batch的train_loss
        train_acc_list = []  # 存放每个batch的train_accuracy
        for X, y in data_iter:
            # y = tf.reshape(y, (y.shape[0],1))  # this step is very important
            with tf.GradientTape() as tape:
                y_pred = model(X)
                l = loss(y_true=tf.one_hot(tf.cast(y, tf.int32), depth=data.num_of_label), y_pred=y_pred)
            grads = tape.gradient(l, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            train_loss_list.append(l)
            train_acc_list.append(evaluate(y_true=y, y_pred=model(X)))
        train_loss = np.mean(train_loss_list)
        train_acc = np.mean(train_acc_list)
        test_loss = loss(y_pred=model(data.test_features),
                         y_true=tf.one_hot(tf.cast(data.test_label, tf.int32), depth=data.num_of_label))
        test_acc = evaluate(y_true=data.test_label, y_pred=model(data.test_features))
        print("epoch:{}, train_loss:{:.5f}, train_acc:{:.3f}, test_loss:{:.5f}, test_acc:{:.3f}".format(epoch + 1,
                                                                                                        train_loss,
                                                                                                        train_acc,
                                                                                                        test_loss.numpy(),
                                                                                                        test_acc))

    # 保存模型权重(参数)
    weights = model.get_weights()
    np.savez('./model/nn_model_attack.npz', w=weights)
