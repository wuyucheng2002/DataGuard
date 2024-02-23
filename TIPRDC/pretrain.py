"""
本模块用于对feature_extractor和classifier进行预训练
"""
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from .data_prepare import Data
from .component import Feature_Extractor, Classifier, evaluate


def train_FE_CF(FE, CF, data, epochs=20, batch_size=256, lr=0.1):
    """
    预训练feature_extractor和classifier
    :param FE: feature_extractor model
    :param CF: classifier model
    :param data: Data()
    :param epochs:
    :param batch_size:
    :param lr:
    :return: (FE, CF)
    """
    data_iter = data.load_array((data.train_features_extractor, data.train_label_extractor), batch_size=batch_size, is_train=True)

    loss = tf.keras.losses.BinaryCrossentropy()
    optimizer_FE = tf.keras.optimizers.SGD(learning_rate=lr)
    optimizer_CF = tf.keras.optimizers.SGD(learning_rate=lr)

    for epoch in range(epochs):
        train_loss_list = []
        train_acc_list = []
        for X, y in tqdm(data_iter):
            with tf.GradientTape(persistent=True) as tape:
                emb = FE(X)
                y_pred = CF(emb)
                l = loss(y_true=tf.one_hot(tf.cast(y, tf.int32), depth=data.num_of_label_u), y_pred=y_pred)
            grads_FE = tape.gradient(l, FE.trainable_variables)
            grads_CF = tape.gradient(l, CF.trainable_variables)
            optimizer_FE.apply_gradients(zip(grads_FE, FE.trainable_variables))
            optimizer_CF.apply_gradients(zip(grads_CF, CF.trainable_variables))

            train_loss_list.append(l)
            train_acc_list.append(evaluate(y_true=y, y_pred=y_pred))
        train_loss = np.mean(train_loss_list)
        train_acc = np.mean(train_acc_list)
        test_loss = loss(y_true=tf.one_hot(tf.cast(data.test_label_extractor, tf.int32), depth=data.num_of_label_u), y_pred=CF(FE(data.test_features_extractor)))
        test_acc = evaluate(y_true=data.test_label_extractor, y_pred=CF(FE(data.test_features_extractor)))

        print("epoch:{}, train_loss:{:.5f}, train_acc:{:.3f}, test_loss:{:.5f}, test_acc:{:.3f}".format(epoch + 1,
                                                                                                        train_loss,
                                                                                                        train_acc,
                                                                                                        test_loss.numpy(),
                                                                                                        test_acc))
    weights_FE = FE.get_weights()
    weights_CF = CF.get_weights()
    np.savez('TIPRDC/model/pre_train/FE.npz', w=weights_FE)
    np.savez('TIPRDC/model/pre_train/CF.npz', w=weights_CF)

    return FE, CF


if __name__ == "__main__":
    FE = Feature_Extractor()
    CF = Classifier()
    data = Data()
    train_FE_CF(FE, CF, data)
