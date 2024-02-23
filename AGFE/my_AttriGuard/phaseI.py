"""
本模块用于执行phaseI
即针对每个噪声空间，找到该空间中下的representative noisy(加噪后的数据)
关于噪声空间，一般是按照label进行划分的；如果拥有10个label，则噪声空间可以划分出10个；当然也可以由用户自己指定；
对于噪声空间i的意义：希望对数据进行加噪使得它们经由分类器的预测结果为i
"""
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from my_AttriGuard.input_data import Data
from my_AttriGuard.model import nn_model_defense


def execute_phaseI(learning_rate, target_label, maxiter=100):
    """
    执行phaseI，寻找各个噪声空间下的representative noisy(加噪后的数据)
    :param learning_rate: nn_defense的学习率
    :param target_label: 噪声空间index，或者说通过加噪(规避攻击)使得分类器的预测值为target_label
    :param maxiter: 加噪过程的最大迭代次数，如果超过这个迭代次数，计算分类器的预测值仍不为target_label也退出循环
    :return: None
    """
    data = Data()

    npz_data = np.load('./model/nn_model_defense.npz', allow_pickle=True)
    weights = npz_data['w']

    model = nn_model_defense(data.num_of_features, data.num_of_label)
    model.build((None, data.num_of_features))

    for i in range(len(model.layers)):
        model.layers[i].trainable = False

    model.set_weights(weights)  # 加载训练好的defense model
    model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.SGD(lr=learning_rate),
                  metrics=['accuracy'])
    model.summary()

    scores = model.evaluate(data.test_features, tf.one_hot(data.test_label, depth=data.num_of_label),
                            verbose=1)  # 查看defense model在加噪前的精度
    print("Test loss: {}".format(scores[0]))
    print("Test accuracy: {}".format(scores[1]))

    print("Target label: {}".format(target_label))

    # model.layers[-2].output是进入sigmoid激活函数前的输出
    # 将[:, target_label]加在这里计算出来gradient_targetlabel才不为0
    dense_model = tf.keras.models.Model(inputs=model.input, outputs=model.layers[-2].output[:, target_label])

    result_array = np.copy(data.test_features)  # 加噪后的数据
    noise_add = 0.0  # 针对整个数据集一共改变了多少个属性值
    for i in tqdm(range(data.test_features.shape[0])):

        sample = result_array[i, :].reshape(1, data.num_of_features)  # 当前样本
        predict_label = np.argmax(model(sample), axis=1)[0]  # defense model对当前样本的加噪前的预测值

        j = 0  # 迭代计数器
        while predict_label != target_label and j < maxiter:
            with tf.GradientTape() as tape:
                x = tf.Variable(sample)
                tape.watch(x)
                dense_output = dense_model(x)
            gradient_targetlabel = tape.gradient(dense_output, x)  # 计算类别概率层输出关于输入的导数，看看改变哪个非敏感属性能够最大概率改变分类器预测值

            max_index = np.argmax((1.0 - sample) * gradient_targetlabel)  # 当改变方式为增加属性值时，最有可能改变分类器预测值的属性
            min_index = np.argmax(-1.0 * sample * gradient_targetlabel)  # 当改变方式为减少属性值时，最有可能改变分类器预测值的属性

            # 选择改变的方式，是选择增加属性值还是减少属性值（此时已经找到各自方式下的最优属性）
            if (1.0 - sample[0, max_index]) * gradient_targetlabel[:, max_index] >= (-1.0) * sample[
                0, min_index] * gradient_targetlabel[:, min_index]:
                sample[0, max_index] = 1.0
            else:
                sample[0, min_index] = 0.0

            predict_label = np.argmax(model(sample), axis=1)[0]
            j += 1

        result_array[i, :] = sample[0, :]
        noise_add += j

    # 对比加载前后的defense分类器的表现
    scores = model.evaluate(data.test_features, tf.one_hot(data.test_label, depth=data.num_of_label), verbose=1)
    print("Before: Test loss: {}, Test accuracy: {}".format(scores[0], scores[1]))
    scores = model.evaluate(result_array, tf.one_hot(data.test_label, depth=data.num_of_label), verbose=1)
    print("After: Test loss: {}, Test accuracy: {}".format(scores[0], scores[1]))
    print("average added noise: {}".format(noise_add / data.num_of_test))

    # 将加噪之后的数据保存起来
    savefilepath = "./data/adversarial_sample/" + str(target_label) + ".txt"
    np.savetxt(savefilepath, result_array)
