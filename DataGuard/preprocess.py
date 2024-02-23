import hashlib
from copy import deepcopy


def md5_encoder(val_list):
    for i in range(len(val_list)):
        val_md5 = hashlib.md5(str(val_list[i]).encode('utf-8'))
        val_list[i] = val_md5.hexdigest()
    return val_list


def get_process_method(df, args, method_name, df_notnull, df_op):
    df = deepcopy(df)
    table_list = [['序号', '属性名', '属性类别', '非缺失值占比', '处理方式']]
    qid_list = []
    cat_dic = {}
    for index, col in enumerate(df.columns):
        if col == args.sensitive:
            cat = '隐私属性'
            op = '删除（用户指定）'
            cat_dic[col] = 'no'
        elif col == args.utility and args.utility != "":
            cat = '效用属性'
            op = '保留'
            cat_dic[col] = 'no'
        elif df_op[index] == 1:
            cat = '其他'
            op = '保留'
            cat_dic[col] = 'no'
        elif df_op[index] == 2:
            cat = '其他'
            op = '删除（行业规范）'
            df.drop(columns=[col], inplace=True)
        elif df_op[index] == 3:
            cat = '主键'
            op = 'MD5加密'
            df[col] = md5_encoder(df[col].tolist())
            cat_dic[col] = 'no'
        elif df_op[index] == 6:
            cat = '其他'
            op = '删除（全空列）'
            df.drop(columns=[col], inplace=True)
        else:
            cat = '准标识符'
            op = method_name
            qid_list.append(col)

            if df_op[index] == 5:
                df[col].fillna('null', inplace=True)
                cat_dic[col] = 'cat'
            elif df_op[index] == 4:
                df[col].fillna(df[col].mean(), inplace=True)
                cat_dic[col] = 'num'
            else:
                raise NotImplementedError

        if len(col) > 16:
            col = col[:16] + '...'
        table_list.append([index + 1, col, cat, df_notnull[index], op])

    # print(df)
    # df.fillna('null', inplace=True)

    qi_index = [list(df.columns).index(qid) for qid in qid_list]

    return table_list, qi_index, qid_list, df, cat_dic



