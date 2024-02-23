import copy

import numpy as np
import pandas as pd
import scipy.spatial.distance as dist
import scipy.io
import matlab.engine

from tqdm import tqdm
from numpy.linalg import norm
from scipy.stats import entropy
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans


def get_DP_obf_X(df, dist_matrix, beta):
    """
    :param df: data to be obfuscated
    :param dist_matrix: distance matrix between users
    :param beta: parameter of differential privacy
    :return: X_obf - obfuscated data, X_ori - original data
    """
    X_ori = {}
    X_obf = {}
    user_size = df.shape[0]
    uid_list = list(df['uid'].values)
    for i in range(user_size):
        user_id = df['uid'][i]
        # get X_ori
        X_ori[user_id] = df[df['uid'] == user_id].values[0, :]

        # get X_obf
        dist_arr = np.array(list(dist_matrix[i]))
        dp_arr = np.exp(-beta*dist_arr)
        prob_list = list(dp_arr / sum(dp_arr))   # compute the swap probabilities
        uidx = np.random.choice(uid_list, 1, p=prob_list)[0]   # randomly choose a user for swap
        X_obf[user_id] = df[df['uid'] == uidx].values[0, :]

    return X_obf, X_ori


def differential_privacy(df_test, beta, repeats=100):
    print("Obfuscation method: DP, beta: {}".format(beta))
    print("generate distance matrix...")
    dist_mat = dist.squareform(dist.pdist(df_test.iloc[:, :-2], 'jaccard'))

    print("start obfuscating...")
    X_obf_dict = {}
    for i in tqdm(range(repeats)):
        X_obf_dict[i], _ = get_DP_obf_X(df_test, dist_mat, beta)
    _, X_ori = get_DP_obf_X(df_test, dist_mat, beta)
    print("obfuscating done.")

    return X_obf_dict, X_ori


def get_random_obf_X(df, p_rand):
    X_ori = {}
    X_obf = {}

    user_size = df.shape[0]
    uid_list = list(df['uid'].values)

    for i in range(user_size):
        obf_flag = np.random.choice([0, 1], 1, p=[0, 1])
        user_id = df['uid'][i]

        # get X_ori
        X_ori[user_id] = df[df['uid']==user_id].values[0, :]

        if obf_flag == 1:
            # get X_obf
            flag = np.random.choice([0, 1], 1, p=[1-p_rand, p_rand])[0]
            if flag == 0:
                X_obf[user_id] = df[df['uid']==user_id].values[0, :]
            else:
                ul = [user_id]
                uidx = np.random.choice(list(set(uid_list) - set(ul)), 1)[0]
                X_obf[user_id] = df[df['uid']==uidx].values[0, :]
        else:
            X_obf[user_id] = df[df['uid']==user_id].values[0, :]

    return X_obf, X_ori


def random_obf(df_test, p_rand, repeats=100):
    print("Obfuscation method: Random, p_rand: {}".format(p_rand))
    print("start obfuscating...")
    X_obf_dict = {}
    for i in tqdm(range(repeats)):
        X_obf_dict[i], _ = get_random_obf_X(df_test, p_rand)
    _, X_ori = get_random_obf_X(df_test, p_rand)
    print("obfuscating done.")

    return X_obf_dict, X_ori


def get_frapp_obf_X(df, gamma):
    X_ori = {}
    X_obf = {}

    user_size = df.shape[0]
    uid_list = list(df['uid'].values)

    e = 1 / (gamma + user_size - 1)
    pfrapp = gamma * e

    for i in range(user_size):
        user_id = df['uid'][i]
        # get X_ori
        X_ori[user_id] = df[df['uid'] == user_id].values[0, :]
        # obf_flag = np.random.choice([0, 1], 1, p=pp)
        p_list = [e] * user_size
        p_list[uid_list.index(user_id)] = pfrapp
        # get X_obf
        uidx = np.random.choice(uid_list, 1, p=p_list)[0]
        X_obf[user_id] = df[df['uid'] == uidx].values[0, :]

    return X_obf, X_ori


def frapp_obf(df_test, gamma, repeats=100):
    print("Obfuscation method: Frapp, gamma: {}".format(gamma))
    print("start obfuscating...")
    X_obf_dict = {}
    for i in tqdm(range(repeats)):
        X_obf_dict[i], _ = get_frapp_obf_X(df_test, gamma)
    _, X_ori = get_frapp_obf_X(df_test, gamma)
    print("obfuscating done.")

    return X_obf_dict, X_ori


def get_similarity_obf_X(sim_mat, df, p):
    sim_mat_copy = copy.deepcopy(sim_mat)

    user_size = df.shape[0]
    uid_list = list(df['uid'].values)
    prob_dict = {}

    # get obfuscation probability
    for i in range(user_size):
        sim_array = sim_mat_copy[i]
        sim_array[i] = 0
        middle_sim = sorted(sim_array)[int(0.8 * len(sim_array))]
        sim_array[sim_array > middle_sim] = 0
        prob_dict[i] = sim_array/sum(sim_array)

    X_ori = {}
    X_obf = {}

    obf_flag = np.random.choice([0, 1], 1, p=[1-p, p])

    for i in range(user_size):
        user_id = df['uid'][i]
        # get X_ori
        X_ori[user_id] = df[df['uid']==user_id].values[0, :]
        # get X_obf
        if obf_flag == 1:
            uidx = np.random.choice(uid_list, 1, p=prob_dict[i])[0]
            X_obf[user_id] = df[df['uid']==uidx].values[0, :]
        else:
            X_obf[user_id] = df[df['uid']==user_id].values[0, :]

    return X_obf, X_ori


def sim_obf(df_test, p, repeats):
    print("Obfuscation method: Similarity, percentage: {}".format(p))
    X_obf_dict = {}
    print("start obfuscating...")
    # get similarity matrix
    itemCols = df_test.columns[:-2]
    df_items = df_test[itemCols]
    sim_mat = cosine_similarity(df_items.values)
    # obfuscation
    for i in tqdm(range(repeats)):
        X_obf_dict[i], _ = get_similarity_obf_X(sim_mat, df_test, p)
    _, X_ori = get_similarity_obf_X(sim_mat, df_test, p)
    print("obfuscating done.")

    return X_obf_dict, X_ori


def get_obf_X(df_cluster, xpgg):
    user_cluster_dict = {}
    cluster_vec = {}
    user_size = df_cluster.shape[0]
    for i in range(user_size):
        user_id = df_cluster['uid'][i]
        cluster_id = df_cluster['cluster'][i]
        user_cluster_dict[user_id] = cluster_id
        if cluster_id in cluster_vec:
            cluster_vec[cluster_id].append(user_id)
        else:
            cluster_vec[cluster_id] = [user_id]

    cluster_size = len(cluster_vec)

    X_obf = {}
    X_ori = {}

    xpgg[xpgg < 0.00001] = 0
    xpgg_norm = normalize(xpgg, axis=0, norm='l1')
    print("obfuscating...")
    for i in range(user_size):
        user_id = df_cluster['uid'][i]
        u_cluster = df_cluster['cluster'][i]
        X_ori[user_id] = df_cluster[df_cluster['uid'] == user_id].values[0, :-1]
        # selecting one cluster to change
        while True:
            change_index = np.random.choice(cluster_size, 1, p=xpgg_norm[:,u_cluster-1])[0]
            change_cluster_index = int(change_index) + 1
            potential_users = cluster_vec[change_cluster_index]
            if len(potential_users) > 0: # potential users may be empty by a slight probability
                break
            else:
                print("not find potential users, re-pick")
        uidx = np.random.choice(potential_users, 1)[0]
        X_obf[user_id] = df_cluster[df_cluster['uid'] == uidx].values[0, :-1]

    return X_obf, X_ori


def PrivCheck(df_test, age_list, deltaX, cluster_num=5, repeats=100):
    print("Obfuscation method: PrivCheck, deltaX: {}".format(deltaX))
    # clustering
    df_cluster = Kmeans_clustering(df_test, cluster_num)

    # solve the obfuscation probability matrix
    # 1. solve PGY matrix
    PGY_Mat = cal_pgy_Matrix(df_cluster, cluster_num, age_list)
    pd.DataFrame(PGY_Mat).to_csv('tmp/pgy_privcheck.csv', index=False, header=None)
    # 2. solve JSD matrix
    JSD_Mat = cal_JSD_Matrix(df_cluster, cluster_num)
    scipy.io.savemat('tmp/JSDM_privcheck.mat', {"JSD_Mat_input": JSD_Mat})
    pd.DataFrame(JSD_Mat).to_csv('tmp/jsd_privcheck.csv', index=False, header=None)

    eng = matlab.engine.start_matlab()
    eng.edit('matlab/PrivCheck', nargout=0)
    # eng.cd('matlab/age_tradeoff_scenario_I', nargout=0)
    xpgg, distortion_budget = np.array(eng.PrivCheck(deltaX, nargout=2))
    xpgg = np.array(xpgg)

    # obfuscation
    X_obf_dict = {}
    for i in tqdm(range(repeats)):
        X_obf_dict[i], _ = get_obf_X(df_test, xpgg)
    _, X_ori = get_obf_X(df_test, xpgg)

    return X_obf_dict, X_ori


def Kmeans_clustering(df_test, cluster_num):
    kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(df_test.values[:, :-2])
    P = kmeans.labels_
    df_cluster = df_test.copy()
    df_cluster['cluster'] = P + 1
    return df_cluster


def cal_pgy_Matrix(df, cluster_dim, age_list):
    user_size = df.shape[0]

    df_dict = {}
    for i in range(1, cluster_dim + 1):
        df_dict[i] = df.loc[df['cluster'] == i, ['age', 'cluster']]

    pgy_Mat = np.zeros((len(age_list), cluster_dim))

    for i in range(1, cluster_dim + 1):
        if df_dict[i].empty:
            for age in age_list:
                age_j = age - age_list[0]
                pgy_Mat[age_j, i - 1] = 0.0000001
        else:
            group_age_cnt = df_dict[i].groupby(['age']).size().reset_index(name='count')
            for age in group_age_cnt['age']:
                age_j = age - age_list[0]
                pgy_Mat[age_j, i - 1] = group_age_cnt.loc[group_age_cnt['age'] == age, 'count'] / user_size

    return pgy_Mat


def JSD(P, Q):
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))


def cal_JSD_Matrix(df_cluster, cluster_dim):
    df_cluster_dict = {}
    for i in range(1, cluster_dim + 1):
        df_cluster_dict[i] = df_cluster.loc[df_cluster['cluster'] == i]

    default_max_JSD = 1
    JSD_Mat = np.ones((cluster_dim, cluster_dim)) * default_max_JSD

    JSD_cluster = []
    for i in range(1, cluster_dim + 1):
        if df_cluster_dict[i].empty:
            continue
        else:
            items_array_i = df_cluster_dict[i].values[:, :-3]
            for j in range(i, cluster_dim + 1):
                if df_cluster_dict[j].empty:
                    continue
                else:
                    items_array_j = df_cluster_dict[j].values[:, :-3]
                    for P in items_array_i:
                        for Q in items_array_j:
                            ##
                            if norm(P, ord=1) == 0 or norm(Q, ord=1)== 0:
                                JSD_cluster.append(1)
                            else:
                                JSD_cluster.append(JSD(P, Q))
                    JSD_Mat[i - 1, j - 1] = np.mean(np.array(JSD_cluster))
                    JSD_Mat[j - 1, i - 1] = np.mean(np.array(JSD_cluster))
                    del JSD_cluster[:]

    return JSD_Mat