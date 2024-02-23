import numpy as np


class Preserver:
    def __init__(self, df, feature_columns, sensitive_column, cat_dic):
        self.df = df
        self.rnum = df.shape[0]
        self.feature_columns = feature_columns
        # self.sensitive_column = sensitive_column
        self.cat_dic = cat_dic
        self.modrian = Mondrian(df, feature_columns, sensitive_column, cat_dic)
        self.k = k, self.partitions, self.ndf = None, None, None

    def anonymize(self, k=3, l=0, p=0.0):
        self.k = k
        self.partitions = self.modrian.partition(k, l, p)
        # print(self.partitions)
        self.ndf = anonymize(self.modrian.df, self.partitions, self.modrian.cat_dic)
        return self.ndf

    # def count_anonymity(self):
    #     return count_anonymity(
    #         self.df,
    #         self.partitions,
    #         self.sensitive_column,
    #         self.cat_dic
    #     )

    def get_eq(self, df):
        df = df.copy(deep=True)
        df['count'] = 1
        return df.groupby(self.feature_columns).count()['count'].values.tolist()

    def utility_scores(self):
        # df_qi = self.df[self.feature_columns].copy(deep=True)
        ndf_qi = self.ndf[self.feature_columns].copy(deep=True)
        gens = []
        for column in self.feature_columns:
            if self.cat_dic[column] == 'cat':
                ndf_qi[column] = ndf_qi[column].map(lambda x: len(str(x).split(',')))
                span = len(self.df[column].unique())
            else:
                ndf_qi[column] = ndf_qi[column].map(
                    lambda x: float(x.split('~')[1]) - float(x.split('~')[0]) if '~' in str(x) else 0.)
                span = self.df[column].max() - self.df[column].min()

            gens.append(ndf_qi[column].mean() / span)

        gen = np.mean(gens)

        neqs = self.get_eq(ndf_qi)
        ndm = np.sum([eq * eq if eq >= self.k else eq * self.rnum for eq in neqs]) / self.rnum

        ncavg = self.rnum / self.k / len(neqs)

        return round(gen, 2), round(ndm, 2), round(ncavg, 2)

    def risk_scores(self, df):
        df_qi = df[self.feature_columns].copy(deep=True)
        eqs = self.get_eq(df_qi)
        ra = np.sum([eq if eq < 5 else 0 for eq in eqs]) / self.rnum
        rb = 1 / min(eqs)
        rc = len(eqs) / self.rnum
        eq_ma, eq_mi = max(eqs), min(eqs)
        r_low = np.sum([eq if eq == eq_ma else 0 for eq in eqs]) / self.rnum
        r_high = np.sum([eq if eq == eq_mi else 0 for eq in eqs]) / self.rnum
        uni = np.sum([eq if eq == 1 else 0 for eq in eqs]) / self.rnum
        return round(ra, 2), round(rb, 2), round(rc, 2), round(r_low, 2), round(r_high, 2), round(uni, 2)


def agg_categorical_column(series):
    # this is workaround for dtype bug of series
    series.astype("category")
    l = [str(n) for n in set(series)]
    return ",".join(l)


def agg_numerical_column(series):
    # return [series.mean()]
    minimum = series.min()
    maximum = series.max()
    if maximum == minimum:
        string = str(maximum)
    else:
        string = f"{minimum}~{maximum}"
    return string


def anonymize(df, partitions, cat_dic, max_partitions=None):
    aggregations = {}
    for column in df.columns:
        # if df[column].dtype.name == "category":
        if cat_dic[column] == 'cat':
            aggregations[column] = agg_categorical_column
        elif cat_dic[column] == 'num':
            aggregations[column] = agg_numerical_column
    ndf = df.copy(deep=True)
    for i, partition in enumerate(partitions):
        if max_partitions is not None and i > max_partitions:
            break
        grouped_columns = df.loc[partition].agg(aggregations, squeeze=False)
        # print(grouped_columns.values)
        # print(partition)
        # print(ndf.loc[partition, list(grouped_columns.index)])
        # ndf.loc[partition, list(grouped_columns.index)] = grouped_columns.values
        ndf.loc[partition, list(grouped_columns.index)] = [grouped_columns.values for _ in partition]
        # print(ndf.loc[partition, list(grouped_columns.index)])
        # print()
    return ndf


# def count_anonymity(df, partitions, sensitive_column, cat_dic, max_partitions=None):
#     aggregations = {}
#     for column in df.columns:
#         # if df[column].dtype.name == "category":
#         if cat_dic[column] == 'cat':
#             aggregations[column] = agg_categorical_column
#         elif cat_dic[column] == 'num':
#             aggregations[column] = agg_numerical_column
#     aggregations[sensitive_column] = "count"
#     rows = []
#     for i, partition in enumerate(partitions):
#         if max_partitions is not None and i > max_partitions:
#             break
#         grouped_columns = df.loc[partition].agg(aggregations, squeeze=False)
#         values = grouped_columns.to_dict()
#         rows.append(values)
#     return rows


class Mondrian:
    def __init__(self, df, feature_columns, sensitive_column, cat_dic):
        self.df = df
        self.feature_columns = feature_columns
        self.sensitive_column = sensitive_column
        self.cat_dic = cat_dic

    def is_valid(self, partition, k=2, l=0, p=0.0):
        # k-anonymous
        if not is_k_anonymous(partition, k):
            return False
        # l-diverse
        if l > 0 and self.sensitive_column is not None:
            diverse = is_l_diverse(
                self.df, partition, self.sensitive_column, l
            )
            if not diverse:
                return False
        # t-close
        if p > 0.0 and self.sensitive_column is not None:
            global_freqs = get_global_freq(self.df, self.sensitive_column)
            close = is_t_close(
                self.df, partition, self.sensitive_column, global_freqs, p
            )
            if not close:
                return False

        return True

    def get_spans(self, partition, scale=None):
        spans = {}
        for column in self.feature_columns:
            # if self.df[column].dtype.name == "category":
            if self.cat_dic[column] == 'cat':
                span = len(self.df[column][partition].unique())
            else:
                span = (
                    self.df[column][partition].max() - self.df[column][partition].min()
                )
            if scale is not None:
                span = span / scale[column]
            spans[column] = span
        return spans

    def split(self, column, partition):
        dfp = self.df[column][partition]
        # if dfp.dtype.name == "category":
        if self.cat_dic[column] == 'cat':
            values = dfp.unique()
            lv = set(values[: len(values) // 2])
            rv = set(values[len(values) // 2 :])
            return dfp.index[dfp.isin(lv)], dfp.index[dfp.isin(rv)]
        else:
            median = dfp.median()
            dfl = dfp.index[dfp < median]
            dfr = dfp.index[dfp >= median]
            return (dfl, dfr)

    def partition(self, k=3, l=0, p=0.0):
        scale = self.get_spans(self.df.index)

        finished_partitions = []
        partitions = [self.df.index]
        while partitions:
            partition = partitions.pop(0)
            spans = self.get_spans(partition, scale)
            for column, span in sorted(spans.items(), key=lambda x: -x[1]):
                lp, rp = self.split(column, partition)
                if not self.is_valid(lp, k, l, p) or not self.is_valid(rp, k, l, p):
                    continue
                partitions.extend((lp, rp))
                break
            else:
                finished_partitions.append(partition)
        return finished_partitions


def is_k_anonymous(partition, k):
    if len(partition) < k:
        return False
    return True


def is_l_diverse(df, partition, sensitive_column, l):
    diversity = len(df.loc[partition][sensitive_column].unique())
    return diversity >= l


def is_t_close(df, partition, sensitive_column, global_freqs, p):
    total_count = float(len(partition))
    d_max = None
    group_counts = (
        df.loc[partition].groupby(sensitive_column)[sensitive_column].agg("count")
    )
    for value, count in group_counts.to_dict().items():
        p = count / total_count
        d = abs(p - global_freqs[value])
        if d_max is None or d > d_max:
            d_max = d
    return d_max <= p


def get_global_freq(df, sensitive_column):
    global_freqs = {}
    total_count = float(len(df))
    group_counts = df.groupby(sensitive_column)[sensitive_column].agg("count")

    for value, count in group_counts.to_dict().items():
        p = count / total_count
        global_freqs[value] = p
    return global_freqs
