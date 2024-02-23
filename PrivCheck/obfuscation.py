import numpy as np
import pandas as pd
import scipy.io
# import matlab.engine
from tqdm import tqdm
from numpy.linalg import norm
from scipy.stats import entropy
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from copy import deepcopy


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


def PrivCheck(df_test, y_sensitive, deltaX, cluster_num=5, repeats=100):
    print("Obfuscation method: PrivCheck, deltaX: {}".format(deltaX))
    # clustering
    df_cluster = Kmeans_clustering(df_test, cluster_num)

    # solve the obfuscation probability matrix
    # 1. solve PGY matrix
    PGY_Mat = cal_pgy_Matrix(df_cluster, cluster_num, y_sensitive)
    pd.DataFrame(PGY_Mat).to_csv('PrivCheck/tmp/pgy_privcheck.csv', index=False, header=None)
    # 2. solve JSD matrix
    JSD_Mat = cal_JSD_Matrix(df_cluster, cluster_num)
    scipy.io.savemat('PrivCheck/tmp/JSDM_privcheck.mat', {"JSD_Mat_input": JSD_Mat})
    pd.DataFrame(JSD_Mat).to_csv('PrivCheck/tmp/jsd_privcheck.csv', index=False, header=None)

    eng = matlab.engine.start_matlab()
    eng.edit('PrivCheck/matlab/PrivCheck', nargout=0)
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


def cal_pgy_Matrix(df_test, cluster_dim, y_sensitive):
    df = deepcopy(df_test)
    df['_sensitive'] = y_sensitive
    user_size = df.shape[0]
    y_unique = np.unique(y_sensitive)

    df_dict = {}
    for i in range(1, cluster_dim + 1):
        df_dict[i] = df.loc[df['cluster'] == i, ['_sensitive', 'cluster']]

    pgy_Mat = np.zeros((len(y_unique), cluster_dim))

    for i in range(1, cluster_dim + 1):
        if df_dict[i].empty:
            for age in y_unique:
                age_j = age - y_unique[0]
                pgy_Mat[age_j, i - 1] = 0.0000001
        else:
            group_age_cnt = df_dict[i].groupby(['_sensitive']).size().reset_index(name='count')
            for age in group_age_cnt['_sensitive']:
                age_j = age - y_unique[0]
                pgy_Mat[age_j, i - 1] = group_age_cnt.loc[group_age_cnt['_sensitive'] == age, 'count'] / user_size

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