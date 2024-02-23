import argparse
import random

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from utils import load_data, split_data, inference_model_training, recommendation
from obfuscation import differential_privacy, random_obf, frapp_obf, sim_obf, PrivCheck


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--obf', type=str, default='frapp', help="DP, random, frapp, similarity, privcheck")
    parser.add_argument('--beta', default=1, help="parameter of DP, >0")
    parser.add_argument('--p_rand', default=0.5, help="parameter of random, 0~1")
    parser.add_argument('--gamma', default=100, help="parameter of frapp, >0")
    parser.add_argument('--percentage', default=0.5, help="parameter of similarity, 0~1")
    parser.add_argument('--deltaX', default=0.60, help="parameter of privcheck, 0~1")

    parser.add_argument('--cluster_num', default=10, help="cluster number for privcheck")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--repeats', type=int, default=1)
    args = parser.parse_args()

    # random.seed(args.seed)

    # load and split data
    df = load_data()
    df_train, df_test, df_test_rec_items = split_data(df)

    # inference model training
    model_rf = inference_model_training(df_train, 'rf')
    model_xgb = inference_model_training(df_train, 'xgb')
    print("inference model training over.")

    # obfuscation
    if args.obf == 'DP':
        X_obf_dict, X_ori = differential_privacy(df_test, args.beta, repeats=args.repeats)
    elif args.obf == 'random':
        X_obf_dict, X_ori = random_obf(df_test, p_rand=args.p_rand, repeats=args.repeats)
    elif args.obf == 'frapp':
        X_obf_dict, X_ori = frapp_obf(df_test, gamma=args.gamma, repeats=args.repeats)
    elif args.obf == 'similarity':
        X_obf_dict, X_ori = sim_obf(df_test, p=args.percentage, repeats=args.repeats)
    elif args.obf == 'privcheck':
        age_list = list(set(df['age'].values))
        age_list.sort()
        X_obf_dict, X_ori = PrivCheck(df_test, age_list, args.deltaX, args.cluster_num, repeats=args.repeats)

    # inference performances & recommendation utility
    rec_oris = []
    rec_obfs = []
    mae_oris_rf = []
    mae_obfs_rf = []
    mae_oris_xgb = []
    mae_obfs_xgb = []

    for i in range(args.repeats):
        # recommendation
        rmse_ori, rmse_obf = recommendation(X_obf_dict[i], X_ori, df_test_rec_items)
        rec_oris.append(rmse_ori)
        rec_obfs.append(rmse_obf)

        # inference
        df_X_obf = pd.DataFrame.from_dict(X_obf_dict[i]).T
        df_x_obf_items = df_X_obf.values[:, :-2]
        df_X_ori = pd.DataFrame.from_dict(X_ori).T
        df_x_ori_items = df_X_ori.values[:, :-2]
        df_x_y = df_test.values[:, -1]

        y_pred_ori_rf = model_rf.predict(df_x_ori_items)
        y_pred_obf_rf = model_rf.predict(df_x_obf_items)
        y_pred_ori_xgb = model_xgb.predict(df_x_ori_items)
        y_pred_obf_xgb = model_xgb.predict(df_x_obf_items)

        mae_ori_rf = mean_absolute_error(df_x_y, y_pred_ori_rf)
        mae_obf_rf = mean_absolute_error(df_x_y, y_pred_obf_rf)
        mae_ori_xgb = mean_absolute_error(df_x_y, y_pred_ori_xgb)
        mae_obf_xgb = mean_absolute_error(df_x_y, y_pred_obf_xgb)

        mae_oris_rf.append(mae_ori_rf)
        mae_obfs_rf.append(mae_obf_rf)
        mae_oris_xgb.append(mae_ori_xgb)
        mae_obfs_xgb.append(mae_obf_xgb)

    m_rmse_ori = np.mean(rec_oris)
    m_rmse_obf = np.mean(rec_obfs)

    m_mae_ori_rf = np.mean(mae_oris_rf)
    m_mae_obf_rf = np.mean(mae_obfs_rf)
    m_mae_ori_xgb = np.mean(mae_oris_xgb)
    m_mae_obf_xgb = np.mean(mae_obfs_xgb)

    print("Utility on Recommendation")
    print("Ori RMSE: {:.4f}, Obf RMSE: {:.4f}".format(m_rmse_ori, m_rmse_obf))
    print("Age Inference Performances")
    print("RandomForest - Ori MAE: {:.4f}, Obf MAE: {:.4f}".format(m_mae_ori_rf, m_mae_obf_rf))
    print("XGBoost - Ori MAE: {:.4f}, Obf MAE: {:.4f}".format(m_mae_ori_xgb, m_mae_obf_xgb))
