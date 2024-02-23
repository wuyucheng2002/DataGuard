import argparse
import numpy as np
import pandas as pd
import json
from DataGuard import *


parser = argparse.ArgumentParser()
parser.add_argument("--upload", type=str, default='')
parser.add_argument("--file", type=str, default='医保_个人基本信息.xlsx')
# parser.add_argument("--file", type=str, default='医疗_居民基本信息表.xlsx')
# parser.add_argument("--file", type=str, default='adult2.xlsx')
parser.add_argument('--type', type=str, default='yb')  # yl, yb, no
# parser.add_argument('--type', type=str, default='yl')  # yl, yb, no
parser.add_argument('--header', type=str, default='yes')  # no, yes
parser.add_argument('--auto', type=str, default='yes')  # no, yes
args = parser.parse_args()

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

if args.type == 'yb':
    dict_name = dict(np.load('DataGuard/yb_name.npy', allow_pickle=True).item())
    df = df.rename(columns=dict_name)
elif args.type == 'yl':
    dict_name = dict(np.load('DataGuard/yl_name.npy', allow_pickle=True).item())
    df = df.rename(columns=dict_name)


if args.header == 'no':
    df_colname, df_notnull, df_op = auto_value(df)
elif args.header == 'yes':
    df_colname, df_notnull, df_op = auto_name(df)
else:
    raise NotImplementedError

if args.auto == 'no':
    df_op = [1 for _ in df_op]

info_dic = {
    "df_colname": list2dict(df_colname),
    "df_notnull": list2dict(df_notnull),
    "df_op": list2str(df_op)
}

with open(args.upload + 'info.json', 'w', encoding='utf-8') as f:
    json.dump(info_dic, f)
