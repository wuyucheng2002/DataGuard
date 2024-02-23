"""
用于控制联合训练并保存结果
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from .data_prepare import Data
from .component import Feature_Extractor, Classifier, MutlInfo
from .train_extractor import train, test_classifier, train_Z, train_classifier


def get_FE(data, epochs=50, batch_size=256, lr=0.1, lbd=0.66):
    """
    联合训练feature extractor, classifier, mutil information estimator
    :param data: Data()
    :param epochs:
    :param batch_size:
    :param lr:
    :param lbd: 用于控制效用与隐私的tradeoff, lbd越大，效用损失越多
    :return: feature extractor
    """
    weights_FE = np.load('TIPRDC/model/pre_train/FE.npz', allow_pickle=True)['w']
    weights_CF = np.load('TIPRDC/model/pre_train/CF.npz', allow_pickle=True)['w']

    FE = Feature_Extractor()
    FE.build((None, data.num_of_features_extractor))
    FE.set_weights(weights_FE)

    CF = Classifier(target_size=data.num_of_label_u)
    CF.build((None, 100))
    CF.set_weights(weights_CF)

    MI = MutlInfo()

    attack_acc_before = test_classifier(FE, CF, data)  # FE未训练成型前attacker CF的攻击准确率

    for epoch in range(epochs):
        print("===========epoch {}===========".format(epoch + 1))
        FE, CF, MI = train(FE, CF, MI, data, batch_size, lr, lbd)
        test_classifier(FE, CF, data)

    attack_acc_after = test_classifier(FE, CF, data)   # FE未训练成型后attacker CF的攻击准确率

    weights_FE = FE.get_weights()
    weights_CF = CF.get_weights()
    weights_MI = MI.get_weights()
    np.savez('TIPRDC/model/all_component/FE.npz', w=weights_FE)
    np.savez('TIPRDC/model/all_component/CF.npz', w=weights_CF)
    np.savez('TIPRDC/model/all_component/MI.npz', w=weights_MI)

    return FE, attack_acc_before, attack_acc_after


def get_ZFE(data, epochs=50, batch_size=256, lr=0.1):
    """
    仅对抗训练feature extractor和classifier
    :param data: Data()
    :param epochs:
    :param batch_size:
    :param lr:
    :return: None
    """
    weights_FE = np.load('TIPRDC/model/pre_train/FE.npz', allow_pickle=True)['w']
    weights_CF = np.load('TIPRDC/model/pre_train/CF.npz', allow_pickle=True)['w']

    FE = Feature_Extractor()
    FE.build((None, data.num_of_features_extractor))
    FE.set_weights(weights_FE)

    CF = Classifier(target_size=data.num_of_label_u)
    CF.build((None, 100))
    CF.set_weights(weights_CF)

    for epoch in range(epochs):
        print("===========epoch {}===========".format(epoch + 1))
        FE, CF = train_Z(FE, CF, data, lr, batch_size)
        test_classifier(FE, CF, data)

    weights_FE = FE.get_weights()
    weights_CF = CF.get_weights()
    np.savez('TIPRDC/model/FE_and_CF/FE.npz', w=weights_FE)
    np.savez('TIPRDC/model/FE_and_CF/CF.npz', w=weights_CF)


def get_classifier(FE_path, data, epochs=50, batch_size=256, lr=0.1):
    """
    仅训练classifier
    :param FE_path: 训练好的feature extractor的保存地址
    :param data: Data()
    :param epochs:
    :param batch_size:
    :param lr:
    :return: None
    """
    weights_FE = np.load(FE_path, allow_pickle=True)['w']
    FE = Feature_Extractor()
    FE.build((None, data.num_of_features_extractor))
    FE.set_weights(weights_FE)

    for i in range(len(FE.layers)):
        FE.layers[i].trainable = False

    CF = Classifier()

    for epoch in range(epochs):
        print("===========epoch {}===========".format(epoch + 1))
        CF = train_classifier(FE, CF, data, lr, batch_size)
        test_classifier(FE, CF, data)

    weights_CF = CF.get_weights()
    np.savez('TIPRDC/model/only_CF/CF.npz', w=weights_CF)


def test_downstream_task(FE, data):
    """
    用于测试extracted feature在下游(target)任务上的表现
    使用简单的逻辑回归作为分类器
    :param FE: feature extractor
    :param data: Data()
    :return: None
    """
    log = LogisticRegression(max_iter=1000)
    X = data.train_features_target
    y = data.train_label_target
    log.fit(X, y)
    y_pred = log.predict(data.test_features_target)
    acc_before = accuracy_score(y_true=data.test_label_target, y_pred=y_pred)
    print("Before Target acc: {}".format(acc_before))

    log = LogisticRegression(max_iter=1000)
    X = FE(data.train_features_target)
    y = data.train_label_target
    log.fit(X, y)
    y_pred = log.predict(FE(data.test_features_target))
    acc_after = accuracy_score(y_true=data.test_label_target, y_pred=y_pred)
    print("After Target acc: {}".format(acc_after))
    # 仅返回测试集上的“新特征”
    return FE(data.test_features_target), acc_before, acc_after


if __name__ == "__main__":
    data = Data()
    FE = get_FE(data)
    test_downstream_task(FE, data)
