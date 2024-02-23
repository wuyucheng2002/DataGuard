import random

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


def load_data():
    df_item_age_uid = pd.read_csv('data/movielens_rating.csv')
    df_item_age_dropU = df_item_age_uid.drop(['uid'], axis=1)
    df_item_age = df_item_age_dropU.groupby(['age']).sum()
    df_item_age.loc['sum_user'] = df_item_age.sum()

    # drop the item has very limited users
    df_item_age_dropI = df_item_age.drop(df_item_age.columns[df_item_age.apply(lambda col: col.sum() < 4)], axis=1)
    df_item_age_uid_dropI = df_item_age_uid.drop(df_item_age.columns[df_item_age.apply(lambda col: ((col != 0).astype(int).sum()) < 3)], axis=1)
    df_item_age_uid_dropI = df_item_age_uid_dropI[df_item_age_uid_dropI.age > 17]
    df_item_age_uid_dropI = df_item_age_uid_dropI[df_item_age_uid_dropI.age < 51]

    df_item_age_dropI.to_csv('data/movie_ages.csv', index=True)
    df_item_age_uid_dropI.to_csv('data/movie_ages_uid.csv', index=False)
    return df_item_age_uid_dropI.reset_index(drop=True)


def split_data(df):
    items = list(df)
    items.remove('age')
    items.remove('uid')
    random.shuffle(items)

    items_train = items[:int(len(items) * 0.8)]
    items_test = list(set(items) - set(items_train))
    df_train_items = df.drop(items_test, axis=1)
    df_test_items = df.drop(items_train, axis=1)

    idx = list(df.index)
    random.shuffle(idx)
    train_idx_list = idx[:int(len(idx)*0.5)]
    test_idx_list = list(set(idx) - set(train_idx_list))

    df_train = df_train_items.loc[train_idx_list].reset_index(drop=True)
    df_test = df_train_items.loc[test_idx_list].reset_index(drop=True)

    df_test_rec_items = df_test_items.loc[test_idx_list]

    print("all user num: {}".format(len(df)))
    print("split train and test over")
    print("train num {}".format(len(df_train)))
    print("test num {}".format(len(df_test)))
    print("train items {}".format(df_train_items.shape[1]))
    print("test items {}".format(df_test_items.shape[1]))

    return df_train, df_test, df_test_rec_items


def inference_model_training(df_train, model_name):
    if model_name == 'rf':
        model = RandomForestRegressor(n_estimators=100, random_state=0)
    elif model_name == 'xgb':
        model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160, silent=True, random_state=0)
    elif model_name == 'svm':
        model = SVR()
    else:
        raise Exception("Sorry, not implemented model. Please choose rf, xgb or svm.")
    y = list(df_train['age'])
    cols = list(df_train.columns)
    cols.remove('age')
    cols.remove('uid')
    X = df_train[cols].values
    model.fit(X, y)

    return model


def recommendation(X_obf, X_ori, df_test):
    df_X_obf = pd.DataFrame.from_dict(X_obf).T
    df_X_ori = pd.DataFrame.from_dict(X_ori).T
    users = list(X_obf.keys())

    random.seed(10)
    random.shuffle(users)
    user_train = users[:int(len(users) * 0.7)]
    user_test = list(set(users) - set(user_train))
    df_obf_trainUser_KnownItem = df_X_obf.drop(user_test).values
    df_obf_testUser_KnownItem = df_X_obf.drop(user_train).values

    testUser_obf_similarity = cosine_similarity(df_obf_testUser_KnownItem, df_obf_trainUser_KnownItem)
    normed_testUser_obf_similarity = normalize(testUser_obf_similarity, axis=1, norm='l1')

    df_test_n = df_test.set_index('uid')
    df_test_n = df_test_n.drop(['age'], axis=1)

    df_trainUser_RcdItem_df = df_test_n.drop(user_test)
    df_trainUser_RcdItem = df_trainUser_RcdItem_df.values
    df_testUser_RcdItem = df_test_n.drop(user_train).values

    trainUser_RcdItem_rating_or_not = df_trainUser_RcdItem_df.copy()
    trainUser_RcdItem_rating_or_not[trainUser_RcdItem_rating_or_not > 0] = 1
    trainUser_RcdItem_rating_or_not_value = trainUser_RcdItem_rating_or_not.values

    small_number = 0.00000001
    testUser_obf_items = np.matrix(normed_testUser_obf_similarity) * np.matrix(df_trainUser_RcdItem) / (
                np.matrix(normed_testUser_obf_similarity) * np.matrix(
            trainUser_RcdItem_rating_or_not_value) + small_number)

    binary_df_testUser_RcdItem = df_testUser_RcdItem.copy()
    binary_df_testUser_RcdItem[binary_df_testUser_RcdItem > 0] = 1  # get the items rated by test users
    testUser_obf_items = np.array(testUser_obf_items) * np.array(binary_df_testUser_RcdItem)  # element-wise multiply

    row, col = testUser_obf_items.shape
    testUser_obf_items = testUser_obf_items.reshape(row * col, )
    df_testUser_RcdItem = df_testUser_RcdItem.reshape(row * col, )
    rmse_obf = np.sqrt(np.sum((testUser_obf_items - df_testUser_RcdItem) ** 2) / np.count_nonzero(df_testUser_RcdItem))

    df_ori_trainUser_KnownItem = df_X_ori.drop(user_test).values[:, :-2]
    df_ori_testUser_KnownItem = df_X_ori.drop(user_train).values[:, :-2]
    testUser_ori_similarity = cosine_similarity(df_ori_testUser_KnownItem, df_ori_trainUser_KnownItem)
    normed_testUser_ori_similarity = normalize(testUser_ori_similarity, axis=1, norm='l1')

    testUser_ori_items = np.matrix(normed_testUser_ori_similarity) * np.matrix(df_trainUser_RcdItem) / (
                np.matrix(normed_testUser_ori_similarity) * np.matrix(
            trainUser_RcdItem_rating_or_not_value) + small_number)
    testUser_ori_items = np.array(testUser_ori_items) * np.array(binary_df_testUser_RcdItem)  # element-wise multiply

    row, col = testUser_ori_items.shape
    testUser_ori_items = testUser_ori_items.reshape(row * col, )
    rmse_ori = np.sqrt(np.sum((testUser_ori_items - df_testUser_RcdItem) ** 2) / np.count_nonzero(df_testUser_RcdItem))

    return rmse_ori, rmse_obf