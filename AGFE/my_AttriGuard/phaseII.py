"""
执行phaseII，找到添加噪声,或者说添加规避攻击的策略(分布)
针对每一个样本，都有这样一个策略，这个策略是一个len(target_label_list)维的向量，每一维存放添加当前类规避攻击的概率
"""
import warnings
import numpy as np
import cvxpy as cp
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from my_AttriGuard.prior import Defense_Prior
from my_AttriGuard.input_data import Data



def execute_phaseII(p, epinslonarray, target_label_list):
    """
    执行phaseII，找到添加噪声,或者说添加规避攻击的策略(分布)
    针对每一个样本，都有这样一个策略，这个策略是一个len(target_label_list)维的向量，每一维存放添加当前类规避攻击的概率
    :param p: defense想要attack求得的数据标签分布(称为defense的先验)
    :param epinslonarray: utility预算列表, 其中epinslon值越大，说明能够牺牲越多的utility
    :return: None
    """
    warnings.filterwarnings("ignore")

    prior = Defense_Prior(p)
    data = Data()

    noise_array = np.loadtxt("./data/adversarial_sample/noise.txt")  # 各个样本在各类规避攻击下改变的属性值的数量
    prob_array = prior.p
    transfer_array = np.loadtxt('./data/attack_prediction/nn.txt')  # 保存attacker分类器对加噪后数据预测的结果，每一行是一个test_user，每一列是规避攻击为i时，attacker分类器的预测结果
    test_label = data.test_label

    Dimension=len(target_label_list)
    Prior = prob_array           # p
    Q = cp.Variable(Dimension)   # M, 针对单个样本如何添加各类规避攻击的分布，如果一共有m类规避攻击，则Q为m维
    objective = cp.sum(cp.kl_div(Prior, Q))
    obj = cp.Minimize(objective)
    probcountlist = []  # 存放在各utility预算下，每个样本被攻击成功的平均概率
    resList = []  # 存放在各utility预算下，最终加噪完成的数据集

    for epinslon in epinslonarray:
        Q_list = []  # 存放每一个样本的指导添加规避攻击的分布
        probcount = 0.  # 全部样本的“攻击成功”的概率总和；其中对于单个样本，这个概率是使用各类规避攻击“攻击成功”的总和
        print("Utility budget: {}".format(epinslon))
        for ii in tqdm(range(data.num_of_test)):
            true_label = test_label[ii]
            constraints = [Q>=1e-10, cp.sum(Q)==1, noise_array[ii, :]*Q<=epinslon]
            prob = cp.Problem(obj, constraints)
            result = prob.solve(solver=cp.SCS)
            Q_list.append(Q.value)
            for i in range(Dimension):
                if abs(transfer_array[ii, i] - true_label)<0.01:  # 以Q.value[i]概率添加i类规避攻击，而此时attacker攻击正确，则attacker攻击正确率为Q.value[i]
                    probcount += Q.value[i]
        Q_list = np.array(Q_list)
        # print(Q_list)
        np.savetxt("./results/Q_" + str(epinslon) + ".txt", Q_list)
        # print('Precision: {}'.format(probcount / test_label.shape[0]))  # 每个样本被攻击成功的平均概率
        probcountlist.append(probcount / (test_label.shape[0]+0.0))

        # 获取加噪后的数据集
        noisy_list = []  # 将各个噪声空间的噪声放在一个列表里
        res = []  # 存放最后的加噪后的数据集
        for target_label in target_label_list:
            noisy_list.append(np.loadtxt('./data/adversarial_sample/{}.txt'.format(target_label)))
        noisy_list = np.array(noisy_list)
        for i in range(Q_list.shape[0]):
            which_noisy = np.random.choice([j for j in range(Q_list.shape[1])], p=Q_list[i]/np.sum(Q_list[i]))
            res.append(noisy_list[which_noisy, i])
        np.savetxt('./results/res_{}.txt'.format(epinslon), res)
        resList.append(np.array(res))

    print("Utility budget list: {}".format(epinslonarray))
    print("Precision List: {}".format(probcountlist))

    # 生成效用指标
    lr = LogisticRegression(max_iter=1000)
    before_acc = np.mean(cross_val_score(lr, data.test_features, data.test_label, scoring='accuracy', cv=5))
    # print("Before Accuracy: {}".format(np.mean(cross_val_score(lr, data.test_features, data.test_label, scoring='accuracy', cv=5))))
    acc_list = []
    for i, res in enumerate(resList):
        res = np.array(res)
        acc = np.mean(cross_val_score(lr, res, data.test_label, scoring='accuracy', cv=5))
        # print("After Accuracy({}): {}".format(epinslonarray[i], acc))
        acc_list.append(acc)
    return resList, probcountlist, before_acc, acc_list

if __name__ == "__main__":
    # execute_phaseII([0.5, 0.5], [0.3], [0, 1])
    # Q_list = np.loadtxt('./results/Q_5.0.txt')
    # print(np.sum(Q_list[0]))
    res = np.loadtxt('./results/res_0.3.txt')
    print(len(res[0]))