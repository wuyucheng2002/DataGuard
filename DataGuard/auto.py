# 1:其他（保留），2:主键（MD5加密），3:隐私属性（直接删除），4:数值型准标识符，5:类别型准标识符，6:全空列（直接删除）
import json
import re
import pandas as pd


def auto_name(df):
    df_notnull = []
    for col in df.columns:
        # print(col)
        df_notnull.append(round(df[col].notnull().sum() / df.shape[0] * 100, 1))

    dict_op = {}
    with open('DataGuard/attribute.json', 'r', encoding='utf-8') as f:
        attr_dict = json.load(f)

    cat_qids = []
    for k in attr_dict["cat_qid"].keys():
        cat_qids += attr_dict["cat_qid"][k]

    cont_qids = []
    for k in attr_dict["cont_qid"].keys():
        cont_qids += attr_dict["cont_qid"][k]

    for idx, col in enumerate(df.columns):
        dict_op[col] = 1

        for name in attr_dict["delete"]:
            if col in name:
                dict_op[col] = 2
                break
        if dict_op[col] == 2:
            continue

        for name in attr_dict["md5"]:
            if col in name:
                dict_op[col] = 3
                break
        if dict_op[col] == 3:
            continue

        if df_notnull[idx] == 0:
            dict_op[col] = 6
            continue

        for name in cat_qids:
            if col in name:
                dict_op[col] = 5
                break
        if dict_op[col] == 5:
            continue

        for name in cont_qids:
            if col in name:
                dict_op[col] = 4
                if df[col].dtype.name in ['object', 'category']:
                    dict_op[col] = 5
                break

    return list(df.columns), df_notnull, list(dict_op.values())


def auto_value(df):
    df_colname = []
    df_notnull = []
    for col in df.columns:
        df_notnull.append(round(df[col].notnull().sum() / df.shape[0] * 100, 1))

    dict_op = {}
    with open('DataGuard/regex.json', 'r', encoding='utf-8') as f:
        regex_dict = json.load(f)

    weizhi = 1
    for idx, col in enumerate(df.columns):
        dict_op[col] = 1
        if df_notnull[idx] == 0:
            dict_op[col] = 6
            df_colname.append('未知' + str(weizhi))
            weizhi += 1
            continue
        for k in regex_dict:
            if len(df.loc[df[col].apply(lambda s: pd.notnull(s) and re.search(regex_dict[k], str(s)) is not None)]) / df[col].notnull().sum() >= 0.8:
                df_colname.append(k)
                dict_op[col] = 3
                break
        if dict_op[col] == 1:
            df_colname.append('未知' + str(weizhi))

    return df_colname, df_notnull, list(dict_op.values())


def list2dict(lis):
    dic = {}
    for i, li in enumerate(lis):
        dic[str(i)] = str(li)
    return dic


def list2str(lis):
    return ''.join([str(li) for li in lis])


def dict2list(dic):
    lis = []
    for k in dic.keys():
        lis.append(dic[k])
    return lis


def str2list(strs):
    return [int(t) for t in strs]


def strlist2list(slis):
    return [float(t) for t in slis.split(',')]
