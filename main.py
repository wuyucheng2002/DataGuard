from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, PageBreak, ListItem, ListFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib import colors
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import hashlib
from anony import *
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False


def md5_encoder(val_list):
    for i in range(len(val_list)):
        val_md5 = hashlib.md5(str(val_list[i]).encode('utf-8'))
        val_list[i] = val_md5.hexdigest()
    return val_list


parser = argparse.ArgumentParser()
parser.add_argument("--upload", type=str, default='../table/')
parser.add_argument("--download", type=str, default='../table/')
parser.add_argument("--file1", type=str, default='医保_个人基本信息.xlsx')
parser.add_argument("--file2", type=str, default=None)
parser.add_argument('--method', type=str, default='K')
parser.add_argument('--ks', type=str, default='0')
parser.add_argument('--ls', type=str, default='0')
parser.add_argument('--ts', type=str, default='0.0')
parser.add_argument('--target', type=str, default='aka129')
parser.add_argument('--qid', type=str, default='aab001,ake010,akc087,aab020')
parser.add_argument('--type', type=str, default='yb')

args = parser.parse_args()
qid_list = args.qid.split(',')
target = args.target
method = args.method

assert method in ['K', 'L', 'T']

pdfmetrics.registerFont(TTFont('SimSun', 'SimSun.ttf'))
ParagraphStyle.defaults['wordWrap'] = "CJK"

doc = SimpleDocTemplate(args.download + '报告.pdf')

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(fontName='SimSun', name='Song', leading=20, fontSize=12, firstLineIndent=24))
styles.add(ParagraphStyle(fontName='SimSun', name='Song2', leading=20, fontSize=12, firstLineIndent=0))
styles.add(ParagraphStyle(fontName='SimSun', name='S_Title', leading=22, fontSize=20, spaceBefore=10,
                          alignment=TA_CENTER, spaceAfter=10))
styles.add(ParagraphStyle(fontName='SimSun', name='S_Head', leading=22, fontSize=15, spaceBefore=15,
                          alignment=TA_LEFT, spaceAfter=10))

story = [Paragraph("“DataGuard”数据安全报告", styles['S_Title']),
         Paragraph("一、数据集分析", styles['S_Head'])]

if args.type == 'yb':
    dict_op = dict(np.load('yb_dict.npy', allow_pickle=True).item())
    dict_name = dict(np.load('yb_name.npy', allow_pickle=True).item())
    story.append(Paragraph(f"数据集分类：医保数据集", styles['Song']))
    key_attr = 'aac001'
else:
    dict_op = dict(np.load('yl_dict.npy', allow_pickle=True).item())
    dict_name = dict(np.load('yl_name.npy', allow_pickle=True).item())
    story.append(Paragraph(f"数据集分类：医疗数据集", styles['Song']))
    key_attr = 'sfz_bm'

if args.file2 is not None:
    df1 = pd.read_excel(args.upload + args.file1)
    df2 = pd.read_excel(args.upload + args.file2)
    story.append(Paragraph(f"第一张表格名称：{args.file1}，记录条数：{df1.shape[0]}，"
                           f"属性个数：{df1.shape[1]}。", styles['Song']))
    story.append(Paragraph(f"第二张表格名称：{args.file2}，记录条数：{df2.shape[0]}，"
                           f"属性个数：{df2.shape[1]}。", styles['Song']))
    story.append(ListFlowable(
        [ListItem(Paragraph(f"主键名称：{dict_name[key_attr]}", styles['Song2']),
                  leftIndent=48, value='circle', bulletColor='#433849', bulletFontSize=6, bulletOffsetY=-3),
         ListItem(Paragraph(f"半标识符名称：{'、'.join([dict_name[i] for i in qid_list])}", styles['Song2']),
                  leftIndent=48, value='circle', bulletColor='#433849', bulletFontSize=6, bulletOffsetY=-3),
         ListItem(Paragraph(f"隐私属性名称：{dict_name[target]}", styles['Song2']),
                  leftIndent=48, value='circle', bulletColor='#433849', bulletFontSize=6, bulletOffsetY=-3),
         ListItem(Paragraph(f"两张表格重复属性名称："
                            f"{'、'.join([dict_name[i]for i in set(df1.columns) & set(df2.columns)])}", styles['Song2']),
                  leftIndent=48, value='circle', bulletColor='#433849', bulletFontSize=6, bulletOffsetY=-3)
         ], bulletType='bullet'))

    df2 = df2.drop(columns=set(df1.columns) & set(df2.columns) - {key_attr}, axis=1)
    df = pd.merge(df1, df2, on=key_attr)
    story.append(Paragraph(f"合并后记录条数：{df.shape[0]}，合并后去除重复列属性个数：{df.shape[1]}。", styles['Song']))

else:
    df = pd.read_excel(args.upload + args.file1)
    story.append(Paragraph(f"表格名称：{args.file1}，记录条数：{df.shape[0]}，"
                           f"属性个数：{df.shape[1]}。", styles['Song']))
    story.append(ListFlowable(
        [ListItem(Paragraph(f"主键名称：{dict_name[key_attr]}", styles['Song2']),
                  leftIndent=48, value='circle', bulletColor='#433849', bulletFontSize=6, bulletOffsetY=-3),
         ListItem(Paragraph(f"半标识符名称：{'、'.join([dict_name[i] for i in qid_list])}", styles['Song2']),
                  leftIndent=48, value='circle', bulletColor='#433849', bulletFontSize=6, bulletOffsetY=-3),
         ListItem(Paragraph(f"隐私属性名称：{dict_name[target]}", styles['Song2']),
                  leftIndent=48, value='circle', bulletColor='#433849', bulletFontSize=6, bulletOffsetY=-3)
         ], bulletType='bullet'))

df_null = {}
for col in df.columns:
    df_null[col] = df[col].notnull().sum() / df.shape[0]
df_null_sorted = sorted(df_null.items(), key=lambda x: x[1], reverse=True)
table_list = [['序号', '属性编码', '属性名', '属性类别', '非缺失值占比', '处理方式']]
qid_list2 = []
cat_dic = {}
for index, (col, val) in enumerate(df_null_sorted):
    if col == target:
        cat = '敏感属性'
        op = '保留'
        cat_dic[col] = 'no'
    elif dict_op[col] == 2:
        cat = '主键'
        op = 'MD5加密'
        df[col] = md5_encoder(df[col].tolist())
        cat_dic[col] = 'no'
    elif dict_op[col] == 3:
        cat = '其他'
        op = '删除（行业规范）'
        df.drop(columns=[col], inplace=True)
    elif val == 0:
        cat = '其他'
        op = '删除（全空列）'
        df.drop(columns=[col], inplace=True)
    elif col in qid_list:
        cat = '准标识符'
        if method == 'K':
            op = 'k-匿名性'
        elif method == 'L':
            op = 'l-多样性'
        else:
            op = 't-相近性'
        qid_list2.append(col)
        if dict_op[col] == 5:
            df[col].fillna('null', inplace=True)
            df[col] = df[col].astype('category')
            cat_dic[col] = 'cat'
        else:
            df[col].fillna(df[col].mean(), inplace=True)
            cat_dic[col] = 'num'
    else:
        cat = '其他'
        op = '保留'
        cat_dic[col] = 'no'

    # 去除非法空格
    if '删除' not in op:
        val_list = df[col].tolist()
        for i in range(len(val_list)):
            if isinstance(val_list[i], str):
                val_list[i] = val_list[i].strip(' ')
        df[col] = val_list

    table_list.append([index + 1, col, dict_name[col][:16], cat, f"{val * 100:.1f}%", op])

story.append(Paragraph(f"处理后属性个数：{len(df.columns)}，"
                       f"总体非缺失值占比：{(df.notnull().sum().sum()/df.size)*100:.2f}%。", styles['Song']))

df.fillna('null', inplace=True)


# 列在行的前面
tablestyle1 = [
    ('FONTNAME', (0, 0), (-1, -1), 'SimSun'),
    ('FONTSIZE', (0, 0), (-1, -1), 10),
    ('BACKGROUND', (0, 0), (-1, 1), '#eadeae'),  # 设置第一行背景颜色
    ('BACKGROUND', (0, 1), (-1, -1), '#fbf9f1'),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),  # 第一行水平居中
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),  # 所有表格上下居中对齐
    # ('TEXTCOLOR', (0, 0), (-1, -1), colors.darkslategray),  # 设置表格内文字颜色
    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),  # 设置表格框线为grey色，线宽为0.5
]

story.append(Table(table_list, style=tablestyle1, spaceBefore=10, spaceAfter=10))

attrs = list(df.columns)
qi_index = [attrs.index(qi) for qi in qid_list2]

ks = [int(k) for k in args.ks.split(',')]
ls = [int(l) for l in args.ls.split(',')]
ts = [float(t) for t in args.ts.split(',')]
# df.to_excel('ndata.xlsx')

if method == 'K':
    if ks == [0]:
        ks = list(range(2, 21))
    ls = [0]
    ts = [0.0]
elif method == 'L':
    if ks == [0]:
        ks = list(range(5, 16))
    if ls == [0]:
        ls = list(range(0, 4))
    ts = [0.0]
else:
    if ks == [0]:
        ks = list(range(5, 16))
    if ts == [0.0]:
        ts = [i * 0.2 for i in range(0, 4)]
    ls = [0]

results = []
index = 0
for k in ks:
    for l in ls:
        for t in ts:
            index += 1
            print(k, l, t)
            pre = Preserver(df, qid_list2, target, cat_dic)
            ndf = pre.anonymize(k=k, l=l, p=t)
            ndf.to_excel(args.download + 'anonymized_' + str(k) + '_' + str(l) + '_' + str(t) + '.xlsx', index=False)
            gen, dm, cavg = pre.utility_scores()
            if index == 1:
                ra, rb, rc, r_low, r_high, uni = pre.risk_scores(df)
                results.append([index, 0, 0, 0.0, '-', '-', '-', round(ra, 2), round(rb, 2), round(rc, 2),
                                round(r_low, 2), round(r_high, 2), round(uni, 2), '-'])
                index += 1
            ra, rb, rc, r_low, r_high, uni = pre.risk_scores(ndf)
            results.append([index, k, l, t, round(gen, 2), round(dm, 2), round(cavg, 2),
                            round(ra, 2), round(rb, 2), round(rc, 2), round(r_low, 2), round(r_high, 2), round(uni, 2),
                            round(rb * gen * 100, 2)])

story.append(PageBreak())
story.append(Paragraph("二、效果分析", styles['S_Head']))
# story.append(Paragraph(f"数据风险和效用：", styles['Song']))

if method == 'K':
    story.append(Paragraph(f"匿名方法：K-匿名性", styles['Song']))
elif method == 'L':
    story.append(Paragraph(f"匿名方法：L-多样性", styles['Song']))
else:
    story.append(Paragraph(f"匿名方法：T-相近性", styles['Song']))


score_head = [['序号', 'k值', 'l值', 't值', '效用指标', '', '', '风险指标', '', '', '', '', '', '权衡指标'],
              ['', '', '', '', 'GIL', 'DM', 'CAVG', 'Ra', 'Rb', 'Rc', 'low', 'high', 'uni', 'tradeoff']]

# 列在行的前面
tablestyle2 = [
    ('FONTNAME', (0, 0), (-1, -1), 'SimSun'),
    ('FONTSIZE', (0, 0), (-1, -1), 10),
    ('BACKGROUND', (0, 0), (-1, 2), '#bbd0f7'),  # 设置第一行背景颜色
    ('BACKGROUND', (0, 2), (-1, -1), '#f6f8ff'),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),  # 第一行水平居中
    # ('ALIGN', (2, -1), (-1, -1), 'LEFT'),  # 第二行到最后一行左右左对齐
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),  # 所有表格上下居中对齐
    # ('TEXTCOLOR', (0, 0), (-1, -1), colors.darkslategray),  # 设置表格内文字颜色
    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),  # 设置表格框线为grey色，线宽为0.5

    ('SPAN', (4, 0), (6, 0)),
    ('SPAN', (7, 0), (12, 0)),

    ('SPAN', (0, 0), (0, 1)),
    ('SPAN', (1, 0), (1, 1)),
    ('SPAN', (2, 0), (2, 1)),
    ('SPAN', (3, 0), (3, 1))
    ]

story.append(Table(score_head + results, style=tablestyle2, spaceBefore=10, spaceAfter=10))

story.append(Paragraph("1、效用指标："
                       "GIL（generalized information loss，泛化信息损失）越小，表示效用损失越少；"
                       "DM（discernibility metric，分辨力指标）越小，表示匿名效果越好；"
                       "CAVG（average equivalence class size metric，平均等价类指标）越接近于1，"
                       "表示匿名效果越好。", styles['Song']))

story.append(Paragraph("2、风险指标："
                       "Ra（lowest prosecutor risk） 刻画重识别概率大于0.2的数据记录占总体的比例，Ra越小，表示风险越低；"
                       "Rb（highest prosecutor risk） 刻画数据集中最大的重识别概率，Rb越小，表示风险越低；"
                       "Rc（average prosecutor risk） 刻画平均重识别概率，Rc越小，表示风险越低；"
                       "low（records affected by lowest risk） 是达到最低风险的记录占总体的比例，在Rc较低的情况下low越大，表示风险越低；"
                       "high（records affected by highest risk） 是达到最高风险的记录占总体的比例，在Rc较低的情况下high越小，表示风险越低；"
                       "uni（sample uniques） 是等价组大小为1的记录数量占总体的比例，uni越小，表示风险越低。", styles['Song']))

story.append(Paragraph("3、权衡指标："
                       "trade-off（风险与效用权衡指标）是效用指标GIL和风险指标Rb的乘积乘上100，"
                       "trade-off越小，表示风险越低，同时效用损失越小。", styles['Song']))

if index > 2:
    a = [re[-1] for re in results[1:]]
    index = np.argmin(a)
    story.append(Paragraph(f"4、推荐取值："
                           f"k值：{results[index + 1][1]}，l值：{results[index + 1][2]}，t值：{results[index + 1][3]}。", styles['Song']))

doc.build(story)
