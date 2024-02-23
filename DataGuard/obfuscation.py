import copy
import numpy as np
import scipy.spatial.distance as dist
from sklearn.metrics.pairwise import cosine_similarity


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


def differential_privacy(df_test, beta, repeats=1):
    # dist_mat = dist.squareform(dist.pdist(df_test.iloc[:, :-2], 'jaccard'))
    dist_mat = dist.squareform(dist.pdist(df_test, 'jaccard'))
    X_obf_dict = {}
    for i in range(repeats):
        X_obf_dict[i], _ = get_DP_obf_X(df_test, dist_mat, beta)
    _, X_ori = get_DP_obf_X(df_test, dist_mat, beta)
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


def random_obf(df_test, p_rand, repeats=1):
    X_obf_dict = {}
    for i in range(repeats):
        X_obf_dict[i], _ = get_random_obf_X(df_test, p_rand)
    _, X_ori = get_random_obf_X(df_test, p_rand)
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


def frapp_obf(df_test, gamma, repeats=1):
    X_obf_dict = {}
    for i in range(repeats):
        X_obf_dict[i], _ = get_frapp_obf_X(df_test, gamma)
    _, X_ori = get_frapp_obf_X(df_test, gamma)
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
    X_obf_dict = {}
    # get similarity matrix
    # itemCols = df_test.columns[:-2]
    itemCols = df_test.columns
    df_items = df_test[itemCols]
    sim_mat = cosine_similarity(df_items.values)
    # obfuscation
    for i in range(repeats):
        X_obf_dict[i], _ = get_similarity_obf_X(sim_mat, df_test, p)
    _, X_ori = get_similarity_obf_X(sim_mat, df_test, p)
    return X_obf_dict, X_ori

