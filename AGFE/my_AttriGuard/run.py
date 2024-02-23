"""
run all procedure
the way to run "run.py" is to type:
python run.py --target_label_list 0 1 --prior 0.3 0.7 --epinslonarray 0.3 5
in the terminal
"""
import argparse

import numpy as np

from input_data import Data
from nn_defense import train_nn_defense
from nn_attack import train_nn_attack
from phaseI import execute_phaseI
from compute_noise import compute_noise
from transfer import transfer
from phaseII import execute_phaseII
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="获取任务相关参数")
    parse.add_argument("--epochs", type=int, default=200, help="training epochs of nn_attack and nn_defense")
    parse.add_argument("--batch_size", type=int, default=256, help="training batch_size of nn_attack and nn_defense")
    parse.add_argument("--learning_rate", type=float, default=0.5, help="training learning_rate of nn_attack and nn_defense")
    parse.add_argument("--maxiter", type=int, default=100, help="the max num of iteration of Evasive attack in phaseI")
    parse.add_argument("--target_label_list", type=int, default=[], help="list of target label in Evasive attack", nargs='+')
    parse.add_argument("--prior", type=float, default=[], help="defense's prior: the p of each target label", nargs='+')
    parse.add_argument("--epinslonarray", type=float, default=[0.3, 5.0], help="the budget of utility in phaseII", nargs='+')
    parse.add_argument("--u", type=str, default='aac009', help="the attr need to protect")
    parse.add_argument("--target", type=str, default='aac084', help="the target label for the downstream task")
    args = parse.parse_args()

    data = Data()
    # 训练defense和attack的分类器
    train_nn_defense(args.batch_size, args.learning_rate, args.epochs)
    train_nn_attack(args.batch_size, args.learning_rate, args.epochs)

    # 若未指定target_label_list，则依据所有可能label划分噪声空间
    if len(args.target_label_list) == 0:
        target_label_list = [i for i in range(data.num_of_label)]
    else:
        target_label_list = args.target_label_list
    # 执行phaseI
    for target_label in target_label_list:
        execute_phaseI(args.learning_rate, target_label, args.maxiter)

    # 统计加噪情况
    compute_noise(target_label_list)

    # 计算attack model对于各加噪数据集的预测值
    transfer(args.learning_rate, target_label_list)

    # 若未指定prior, 则以均匀分布作为prior
    if len(args.prior) == 0:
        p = [1/data.num_of_label for i in range(data.num_of_label)]
    elif len(args.prior) == len(target_label_list):
        p = args.prior
    else:
        raise Exception("prior的shape与target_label_list的shape不符")

    # 执行phaseII
    resList, probcountlist = execute_phaseII(p, args.epinslonarray, target_label_list)












