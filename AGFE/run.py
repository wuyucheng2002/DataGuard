"""
接收用户参数，并执行属性保护任务
the way to run "run.py" is to type:
python run.py --algorithm=‘AG’
or
python run.py --algorithm='FE'
in the terminal
"""
import argparse
import os

from my_AttriGuard.input_data import Data as Data_AG
from my_AttriGuard.nn_defense import train_nn_defense
from my_AttriGuard.nn_attack import train_nn_attack
from my_AttriGuard.phaseI import execute_phaseI
from my_AttriGuard.compute_noise import compute_noise
from my_AttriGuard.transfer import transfer
from my_AttriGuard.phaseII import execute_phaseII

from my_TIPRDC.data_prepare import Data as Data_TIPRDC
from my_TIPRDC.component import Feature_Extractor, Classifier, MutlInfo
from my_TIPRDC.pretrain import train_FE_CF
from my_TIPRDC.main import get_FE, test_downstream_task

if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="获取用户参数")
    parse.add_argument("--algorithm", type=str, default='AG', help="the privacy algorithm you want to run")
    
    '''AttriGuard参数'''
    # 重要参数
    parse.add_argument("--AG_target_label_list", type=int, default=[], help="list of target label in Evasive attack",
                       nargs='+')
    parse.add_argument("--AG_prior", type=float, default=[], help="defense's prior: the p of each target label",
                       nargs='+')
    parse.add_argument("--AG_epinslonarray", type=float, default=[0.3, 5.0], help="the budget of utility in phaseII",
                       nargs='+')
    # 次要参数
    parse.add_argument("--u", type=str, default='aac009', help="the attr need to protect")
    parse.add_argument("--target", type=str, default='aac084', help="the target label for the downstream task")
    parse.add_argument("--AG_epochs", type=int, default=200, help="training epochs of nn_attack and nn_defense")
    parse.add_argument("--AG_batch_size", type=int, default=256, help="training batch_size of nn_attack and nn_defense")
    parse.add_argument("--AG_learning_rate", type=float, default=0.5,
                       help="training learning_rate of nn_attack and nn_defense")
    parse.add_argument("--AG_maxiter", type=int, default=100, help="the max num of iteration of Evasive attack in phaseI")

    
    '''TIPRDC参数'''
    # 重要参数
    parse.add_argument("--FE_target_label", type=str, default='aac084', help="the label for downstream task")
    parse.add_argument("--FE_u", type=str, default='aac009', help="the private attribute")
    parse.add_argument("--FE_lbd", type=float, default=0.5, help="for controlling the tradeoff between utility and privacy, higher lbd leads to higher privacy")
    # 次要参数
    parse.add_argument("--FE_epochs", type=int, default=50, help="training epochs of components")
    parse.add_argument("--FE_batch_size", type=int, default=256, help="training batch_size of components")
    parse.add_argument("--FE_lr", type=float, default=0.1, help="training learning_rate of components")
    parse.add_argument("--FE_pretrain_epochs", type=int, default=20, help="training epochs of FE and CF for pretraining")
    parse.add_argument("--FE_pretrain_batch_size", type=int, default=256, help="training batch size of FE and CF for pretraining")
    parse.add_argument("--FE_pretrain_lr", type=float, default=0.1, help="training learning rate of FE and CF for pretraining")
    parse.add_argument("--FE_embedding_dim", type=int, default=100, help="the dimension of the features extracted (embedding)")
    parse.add_argument("--FE_hidden_dim", type=int, default=100, help="the dimension of classifier's hidden layer")

    args = parse.parse_args()

    if args.algorithm == "AG":
        # 切换路径
        if 'my_AttriGuard' in os.getcwd():
            pass
        else:
            os.chdir(os.getcwd() + '\\my_AttriGuard')

        '''AttriGuard'''
        data1 = Data_AG()
        # 训练defense和attack的分类器
        train_nn_defense(args.AG_batch_size, args.AG_learning_rate, args.AG_epochs)
        train_nn_attack(args.AG_batch_size, args.AG_learning_rate, args.AG_epochs)
        # 若未指定target_label_list，则依据所有可能label划分噪声空间
        if len(args.AG_target_label_list) == 0:
            target_label_list = [i for i in range(data1.num_of_label)]
        else:
            target_label_list = args.AG_target_label_list
        # 执行phaseI
        for target_label in target_label_list:
            execute_phaseI(args.AG_learning_rate, target_label, args.AG_maxiter)
        # 统计加噪情况
        compute_noise(target_label_list)
        # 计算attack model对于各加噪数据集的预测值
        transfer(args.AG_learning_rate, target_label_list)
        # 若未指定prior, 则以均匀分布作为prior
        if len(args.AG_prior) == 0:
            p = [1 / data1.num_of_label for i in range(data1.num_of_label)]
        elif len(args.AG_prior) == len(target_label_list):
            p = args.AG_prior
        else:
            raise Exception("prior的shape与target_label_list的shape不符")
        # 执行phaseII, res_list是各utility_budget下加噪后的数据集, precision_list是各utility_budget下attack的预测准确率（privacy）, 暂无效用
        res_list, precision_list, before_acc, acc_list = execute_phaseII(p, args.AG_epinslonarray, target_label_list)
        print("Before Accuracy: {}".format(before_acc))
        for i, acc in enumerate(acc_list):
            print("After Accuracy({}): {}".format(args.AG_epinslonarray[i], acc))

    elif args.algorithm == "FE":
        # os.chdir(os.path.dirname(os.getcwd()))
        # 切换路径
        if 'my_TIPRDC' in os.getcwd():
            pass
        else:
            os.chdir(os.getcwd() + '\\my_TIPRDC')

        '''TIPRDC'''
        data2 = Data_TIPRDC(target=args.FE_target_label, u=args.FE_u)
        FE = Feature_Extractor(args.FE_embedding_dim)
        CF = Classifier(target_size=data2.num_of_label_u, hidden_dim=args.FE_hidden_dim)
        # 预训练FE和CF
        train_FE_CF(FE, CF, data2, args.FE_pretrain_epochs, args.FE_pretrain_batch_size, args.FE_pretrain_lr)
        # 获取最终FE，attack_acc_before是FE训练成型前attacker CF的攻击准确率，attack_acc_after是FE训练成型后attacker CF的攻击准确率，反映了privacy
        FE, attack_acc_before, attack_acc_after = get_FE(data2, args.FE_epochs, args.FE_batch_size, args.FE_lr, args.FE_lbd)
        # 测试下游任务完成情况, Embedding是从原特征中提取的具有隐私能力的”新特征“，acc_before是使用原特征的下游任务准确率，acc_after是使用”新特征“的下游任务准确率，反映了效用
        Embedding, acc_before, acc_after = test_downstream_task(FE, data2)
        print("Attack_Acc_Before:{} || Attack_Acc_After:{}".format(attack_acc_before, attack_acc_after))
        print("Acc_Before:{} || Acc_After:{}".format(acc_before, acc_after))
        # print(Embedding.shape)
    else:
        raise Exception("输入算法名称有误！")




