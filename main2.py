import matplotlib.pyplot as plt
import argparse
from DataGuard import *
from PrivCheck import PrivCheck
from AttriGuard import *
from TIPRDC import *
import warnings
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False

# k匿名：k值（正整数，推荐值：3），隐私属性，效用属性
# L多样性：L值（正整数，推荐值：2），隐私属性，效用属性
# T相近性：T值（0~1，推荐值：3），隐私属性，效用属性
# 差分隐私：差分隐私参数（>0，推荐值：1），隐私属性，效用属性
# 混淆（随机）：扰动系数（0~1，推荐值：0.5），隐私属性，效用属性
# 混淆（加权）：扰动系数（正整数，推荐值：20），隐私属性，效用属性
# 混淆（相似度）：扰动系数（0~1，推荐值：0.5），隐私属性，效用属性
# 混淆（PrivCheck）：扰动系数（0~1，推荐值：0.5），隐私属性，效用属性
# 对抗样本：扰动系数（>0，推荐值：1），隐私属性，效用属性
# 对抗训练：权衡系数（0~1，推荐值：0.5），隐私属性，效用属性

method_names = {
    "K": "K匿名",
    "L": "L多样性",
    "T": "T相近性",
    "DP": "差分隐私",
    "random": "混淆（随机）",
    "weight": "混淆（加权）",
    "sim": "混淆（相似度）",
    "obf": "混淆（PrivCheck）",
    "noise": "对抗样本（AttriGuard）",
    "adver": "对抗训练（TIPRDC）",
}


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", type=str, default='')
    parser.add_argument("--download", type=str, default='')
    parser.add_argument("--file", type=str, default='医保_个人基本信息.xlsx')
    # parser.add_argument("--file", type=str, default='医疗_居民基本信息表.xlsx')
    # parser.add_argument("--file", type=str, default='adult2.xlsx')
    parser.add_argument('--method', type=str, default='K')
    # parser.add_argument('--method', type=str, default='random')
    parser.add_argument('--params', type=str, default='2,3,4,5,6,7,8')
    # parser.add_argument('--params', type=str, default='0.2,0.4,0.6,0.8,1')
    parser.add_argument('--sensitive', type=str, default="")
    # parser.add_argument('--sensitive', type=str, default='性别')
    parser.add_argument('--utility', type=str, default="")
    # parser.add_argument('--utility', type=str, default='性别')
    parser.add_argument('--type', type=str, default='yb')  # yl, yb, no
    # parser.add_argument('--type', type=str, default='yl')  # yl, yb, no
    parser.add_argument('--header', type=str, default='yes')  # no, yes

    #AttriGuard参数
    # 重要参数
    parser.add_argument("--AG_epinslonarray", type=float, default=[0.3, 5.0], help="the budget of utility in phaseII", nargs='+')
    parser.add_argument("--AG_target_label_list", type=int, default=[], help="list of target label in Evasive attack", nargs='+')
    parser.add_argument("--AG_prior", type=float, default=[], help="defense's prior: the p of each target label", nargs='+')
    parser.add_argument("--AG_epochs", type=int, default=200, help="training epochs of nn_attack and nn_defense")
    parser.add_argument("--AG_batch_size", type=int, default=256, help="training batch_size of nn_attack and nn_defense")
    parser.add_argument("--AG_learning_rate", type=float, default=0.5, help="training learning_rate of nn_attack and nn_defense")
    parser.add_argument("--AG_maxiter", type=int, default=100, help="the max num of iteration of Evasive attack in phaseI")

    # TIPRDC参数
    parser.add_argument("--FE_lbd", type=float, default=0.5, help="for controlling the tradeoff between utility and privacy, higher lbd leads to higher privacy")
    parser.add_argument("--FE_epochs", type=int, default=50, help="training epochs of components")
    parser.add_argument("--FE_batch_size", type=int, default=256, help="training batch_size of components")
    parser.add_argument("--FE_lr", type=float, default=0.1, help="training learning_rate of components")
    parser.add_argument("--FE_pretrain_epochs", type=int, default=20, help="training epochs of FE and CF for pretraining")
    parser.add_argument("--FE_pretrain_batch_size", type=int, default=256, help="training batch size of FE and CF for pretraining")
    parser.add_argument("--FE_pretrain_lr", type=float, default=0.1, help="training learning rate of FE and CF for pretraining")
    parser.add_argument("--FE_embedding_dim", type=int, default=100, help="the dimension of the features extracted (embedding)")
    parser.add_argument("--FE_hidden_dim", type=int, default=100, help="the dimension of classifier's hidden layer")

    args = parser.parse_args()

    if args.method != "K":
        assert args.sensitive != ""

    return args


args = get_parser()

if args.header == 'no':
    header = None
elif args.header == 'yes':
    header = 0
else:
    raise NotImplementedError

if args.file.split('.')[-1] == 'xlsx':
    df = pd.read_excel(args.upload + args.file, header=header)
elif args.file.split('.')[-1] in ['csv', 'txt']:
    if ';' in pd.read_csv(args.upload + args.file).iat[0, 0]:
        df = pd.read_csv(args.upload + args.file, sep=';', header=header)
    else:
        df = pd.read_csv(args.upload + args.file, header=header)
else:
    raise NotImplementedError

with open(args.upload + 'info.json', 'r', encoding='utf-8') as f:
    info_dic = json.load(f)

df_notnull = dict2list(info_dic["df_notnull"])
# df_op = str2list(args.df_op)
df_op = str2list(info_dic["df_op"])

if args.type == 'yb':
    dict_name = dict(np.load('DataGuard/yb_name.npy', allow_pickle=True).item())
    df = df.rename(columns=dict_name)
elif args.type == 'yl':
    dict_name = dict(np.load('DataGuard/yl_name.npy', allow_pickle=True).item())
    df = df.rename(columns=dict_name)

method_name = method_names[args.method]

table_list, qi_index, qid_list, df2, cat_dic = get_process_method(df, args, method_name, df_notnull, df_op)

results = []
index = 0
info_draw = {"privacy": {}, "utility": {}}

# priv（隐私属性推断攻击准确率）
# dist（距离）
# acc（下游预测任务准确率）

if args.method in ['K', 'L', 'T']:
    info_draw["privacy"]['Ra（最低检察官风险）'] = {}
    info_draw["privacy"]['Rb（最高检察官风险）'] = {}
    info_draw["privacy"]['Rc（平均检察官风险）'] = {}
    info_draw["privacy"]['r_low（最低风险影响的记录数）'] = {}
    info_draw["privacy"]['r_high（最高风险影响的记录数）'] = {}
    info_draw["privacy"]['uni（唯一样本）'] = {}

    info_draw["utility"]['GIL（泛化信息损失）'] = {}
    info_draw["utility"]['DM（分辨力指标）'] = {}
    info_draw["utility"]['CAVG（平均等价类指标）'] = {}

    for param in strlist2list(args.params):
        index += 1
        pre = Preserver(df2, qid_list, args.sensitive, cat_dic)
        if args.method == 'K':
            ndf = pre.anonymize(k=param)
        elif args.method == 'L':
            ndf = pre.anonymize(l=param)
        else:
            ndf = pre.anonymize(p=param)
        ndf.to_excel(args.download + args.file[:-5] + '_' + args.method + '_' + str(param) + '.xlsx', index=False)

        if index == 1:
            ra, rb, rc, r_low, r_high, uni = pre.risk_scores(df2)

            info_draw["utility"]['GIL（泛化信息损失）']['原始'] = 0
            info_draw["utility"]['DM（分辨力指标）']['原始'] = 1
            info_draw["utility"]['CAVG（平均等价类指标）']['原始'] = 1

            info_draw["privacy"]['Ra（最低检察官风险）']['原始'] = ra
            info_draw["privacy"]['Rb（最高检察官风险）']['原始'] = rb
            info_draw["privacy"]['Rc（平均检察官风险）']['原始'] = rc
            info_draw["privacy"]['r_low（最低风险影响的记录数）']['原始'] = r_low
            info_draw["privacy"]['r_high（最高风险影响的记录数）']['原始'] = r_high
            info_draw["privacy"]['uni（唯一样本）']['原始'] = uni

            results.append([index, '原始', 0, 1, 1, ra, rb, rc, r_low, r_high, uni, '-'])
            index += 1

        gen, dm, cavg = pre.utility_scores()
        info_draw["utility"]['GIL（泛化信息损失）'][str(param)] = gen
        info_draw["utility"]['DM（分辨力指标）'][str(param)] = dm
        info_draw["utility"]['CAVG（平均等价类指标）'][str(param)] = cavg

        ra, rb, rc, r_low, r_high, uni = pre.risk_scores(ndf)
        info_draw["privacy"]['Ra（最低检察官风险）'][str(param)] = ra
        info_draw["privacy"]['Rb（最高检察官风险）'][str(param)] = rb
        info_draw["privacy"]['Rc（平均检察官风险）'][str(param)] = rc
        info_draw["privacy"]['r_low（最低风险影响的记录数）'][str(param)] = r_low
        info_draw["privacy"]['r_high（最高风险影响的记录数）'][str(param)] = r_high
        info_draw["privacy"]['uni（唯一样本）'][str(param)] = uni

        results.append([index, param, gen, dm, cavg, ra, rb, rc, r_low, r_high, uni, round(rb * gen, 3)])

        print_pdf(args, df, qid_list, table_list, results, info_draw, method_name)
        with open(args.download + 'info_draw.json', 'w', encoding='utf-8') as f:
            json.dump(info_draw, f)

elif args.method in ['DP', 'random', 'weight', 'sim', 'obf']:
    info_draw["privacy"]['priv（隐私属性推断攻击准确率）'] = {}
    info_draw["utility"]['dist（距离）'] = {}

    df2 = df.fillna(0)
    y_sensitive = df2[args.sensitive].values
    df2.drop(columns=[args.sensitive], inplace=True)
    # print(y_sensitive)

    if args.utility != "":
        y_utility = df2[args.utility].values
        info_draw["utility"]['acc（下游预测任务准确率）'] = {}
        df2.drop(columns=[args.utility], inplace=True)

    for param in strlist2list(args.params):
        df2 = pd.get_dummies(df2)
        index += 1
        df2['uid'] = df2.index
        if args.method == 'DP':
            X_obf_dict, X_ori = differential_privacy(df2, param, repeats=1)
        elif args.method == 'random':
            X_obf_dict, X_ori = random_obf(df2, param, repeats=1)
        elif args.method == 'weight':
            X_obf_dict, X_ori = frapp_obf(df2, param, repeats=1)
        elif args.method == 'sim':
            X_obf_dict, X_ori = sim_obf(df2, param, repeats=1)
        else:
            X_obf_dict, X_ori = PrivCheck(df2, y_sensitive, param, repeats=1)
        X_obf = X_obf_dict[0]

        df_X_obf = pd.DataFrame.from_dict(X_obf).T
        df_X_ori = pd.DataFrame.from_dict(X_ori).T

        if index == 1:
            pr = rf_acc(df_X_ori, y_sensitive)
            dist = 0
            info_draw["privacy"]['priv（隐私属性推断攻击准确率）']['原始'] = pr
            info_draw["utility"]['dist（距离）']['原始'] = dist
            if args.utility != "":
                ut_o = rf_acc(df_X_ori, y_utility)
                info_draw["utility"]['acc（下游预测任务准确率）']['原始'] = ut_o
                results.append([index, '原始', dist, ut_o, pr, '-'])
            else:
                results.append([index, '原始', dist, pr, '-'])
            index += 1

        pr = rf_acc(df_X_obf, y_sensitive)
        dist = cal_dist(df_X_ori, df_X_obf)
        info_draw["privacy"]['priv（隐私属性推断攻击准确率）'][str(param)] = pr
        info_draw["utility"]['dist（距离）'][str(param)] = dist
        if args.utility != "":
            ut = rf_acc(df_X_obf, y_utility)
            info_draw["utility"]['acc（下游预测任务准确率）'][str(param)] = ut
            results.append([index, param, dist, ut, pr, round((ut_o - ut) * pr, 3)])
        else:
            results.append([index, param, dist, pr, round(dist * pr, 3)])

        if args.utility != "":
            df_X_obf[args.utility] = y_utility
        df_X_obf.to_excel(args.download + args.file[:-5] + '_' + args.method + '_' + str(param) + '.xlsx', index=False)

        print_pdf(args, df, qid_list, table_list, results, info_draw, method_name)
        with open(args.download + 'info_draw.json', 'w', encoding='utf-8') as f:
            json.dump(info_draw, f)

elif args.method == "noise":
    data1 = Data_AG(df2, args.utility, args.sensitive)
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
    res_list, precision_list, before_acc, acc_list = execute_phaseII(p, args.params, target_label_list)
    print("Before Accuracy: {}".format(before_acc))
    for i, acc in enumerate(acc_list):
        print("After Accuracy({}): {}".format(args.AG_epinslonarray[i], acc))

elif args.method == "adver":
    data2 = Data_TIPRDC(df, args.utility, args.sensitive)
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
    raise NotImplementedError

