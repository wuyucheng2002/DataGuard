from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, PageBreak, ListItem, ListFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib import colors
import numpy as np


def print_pdf(args, df, qid_list, table_list, results, info_draw, method_name):
    pdfmetrics.registerFont(TTFont('SimSun', 'DataGuard/SimSun.ttf'))
    ParagraphStyle.defaults['wordWrap'] = "CJK"

    doc = SimpleDocTemplate(args.download + '报告_' + args.method + '.pdf')

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(fontName='SimSun', name='Song', leading=20, fontSize=12, firstLineIndent=24))
    styles.add(ParagraphStyle(fontName='SimSun', name='Song2', leading=20, fontSize=12, firstLineIndent=0))
    styles.add(ParagraphStyle(fontName='SimSun', name='S_Title', leading=22, fontSize=20, spaceBefore=10,
                              alignment=TA_CENTER, spaceAfter=10))
    styles.add(ParagraphStyle(fontName='SimSun', name='S_Head', leading=22, fontSize=15, spaceBefore=15,
                              alignment=TA_LEFT, spaceAfter=10))

    story = list()

    story.append(Paragraph("“DataGuard”数据安全报告", styles['S_Title']))
    story.append(Paragraph("一、数据集分析", styles['S_Head']))

    story.append(Paragraph(f"表格名称：{args.file}，记录条数：{df.shape[0]}，"
                           f"属性个数：{df.shape[1]}。", styles['Song']))
    story.append(ListFlowable(
        [ListItem(Paragraph(f"准标识符名称：{'、'.join(qid_list)}", styles['Song2']),
                  leftIndent=48, value='circle', bulletColor='#433849', bulletFontSize=6, bulletOffsetY=-3),
         ListItem(Paragraph(f"隐私属性名称：{args.sensitive if args.sensitive != '' else '无'}", styles['Song2']),
                  leftIndent=48, value='circle', bulletColor='#433849', bulletFontSize=6, bulletOffsetY=-3),
         ListItem(Paragraph(f"效用属性名称：{args.utility if args.utility != '' else '无'}", styles['Song2']),
                  leftIndent=48, value='circle', bulletColor='#433849', bulletFontSize=6, bulletOffsetY=-3)
         ], bulletType='bullet'))

    story.append(Paragraph(f"处理后属性个数：{len(df.columns)}，"
                           f"总体非缺失值占比：{(df.notnull().sum().sum() / df.size) * 100:.2f}%。", styles['Song']))

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

    story.append(PageBreak())
    story.append(Paragraph("二、效果分析", styles['S_Head']))
    # story.append(Paragraph(f"数据风险和效用：", styles['Song']))

    story.append(Paragraph(f"匿名方法：" + method_name, styles['Song']))

    head1 = ['序号', '参数值']
    head2 = ['', '']
    ut_num = 0
    pr_num = 0

    for idx, ut in enumerate(info_draw['utility']):
        ut_num += 1
        if idx == 0:
            head1.append('效用指标')
        else:
            head1.append('')
        head2.append(ut.split('（')[0])

    for idx, pr in enumerate(info_draw['privacy']):
        pr_num += 1
        if idx == 0:
            head1.append('风险指标')
        else:
            head1.append('')
        head2.append(pr.split('（')[0])

    head1.append('权衡指标')
    head2.append('tradeoff')

    score_head = [head1, head2]
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
        ('SPAN', (0, 0), (0, 1)),
        ('SPAN', (1, 0), (1, 1)),
        ('SPAN', (2, 0), (2 + ut_num - 1, 0)),
        ('SPAN', (2 + ut_num, 0), (2 + ut_num + pr_num - 1, 0))
    ]

    story.append(Table(score_head + results, style=tablestyle2, spaceBefore=10, spaceAfter=10))

    a = [re[-1] for re in results[1:]]
    story.append(Paragraph(f"推荐取值：参数值 = {results[np.argmin(a) + 1][1]}。", styles['Song']))

    story.append(Paragraph("\0", styles['Song']))
    story.append(Paragraph("1、效用指标：", styles['Song']))
    story.append(ListFlowable(
        [ListItem(Paragraph("GIL（generalized information loss，泛化信息损失）通过量化已泛化的域值的比例，捕获泛化特定属性时产生的损失。GIL越小，表示效用损失越少。", styles['Song2']),
                  leftIndent=48, value='circle', bulletColor='#433849', bulletFontSize=6, bulletOffsetY=-3),
         ListItem(Paragraph("DM（discernibility metric，分辨力指标）通过给每个记录分配一个惩罚值，来衡量一个记录与其他记录的不可区分程度，惩罚值等于它所属的等价类的大小。DM越小，表示效用损失越少。", styles['Song2']),
                  leftIndent=48, value='circle', bulletColor='#433849', bulletFontSize=6, bulletOffsetY=-3),
         ListItem(Paragraph("CAVG（average equivalence class size metric，平均等价类指标）测量等价组的创建是否接近最佳情况。CAVG越小，表示效用损失越少。", styles['Song2']),
                  leftIndent=48, value='circle', bulletColor='#433849', bulletFontSize=6, bulletOffsetY=-3),
         ListItem(Paragraph("dist（distance，距离）衡量原始数据集和新数据集的平均欧式距离。dist越小，表示效用损失越少。", styles['Song2']),
                  leftIndent=48, value='circle', bulletColor='#433849', bulletFontSize=6, bulletOffsetY=-3),
         ListItem(Paragraph("acc（downstream prediction task accuracy，下游预测任务准确率）基于隐私处理后的数据集训练下游任务的机器学习模型，计算下游任务分类准确率（此指标的计算需指定效用属性）。acc越高，表示效用损失越少。", styles['Song2']),
                  leftIndent=48, value='circle', bulletColor='#433849', bulletFontSize=6, bulletOffsetY=-3),
         ], bulletType='bullet'))

    story.append(Paragraph("\0", styles['Song']))
    story.append(Paragraph("2、风险指标：", styles['Song']))
    story.append(ListFlowable(
        [ListItem(Paragraph("Ra（lowest prosecutor risk，最低检察官风险）刻画重识别概率大于0.2的数据记录占总体的比例，Ra越小，表示风险越低。", styles['Song2']),
                  leftIndent=48, value='circle', bulletColor='#433849', bulletFontSize=6, bulletOffsetY=-3),
         ListItem(Paragraph("Rb（highest prosecutor risk，最高检察官风险）刻画数据集中最大的重识别概率，Rb越小，表示风险越低。", styles['Song2']),
                  leftIndent=48, value='circle', bulletColor='#433849', bulletFontSize=6, bulletOffsetY=-3),
         ListItem(Paragraph("Rc（average prosecutor risk ，平均检察官风险）刻画平均重识别概率，Rc越小，表示风险越低。", styles['Song2']),
                  leftIndent=48, value='circle', bulletColor='#433849', bulletFontSize=6, bulletOffsetY=-3),
         ListItem(Paragraph("low（records affected by lowest risk，最低风险影响的记录数）是达到最低风险的记录占总体的比例，在Rc较低的情况下low越大，表示风险越低。", styles['Song2']),
                  leftIndent=48, value='circle', bulletColor='#433849', bulletFontSize=6, bulletOffsetY=-3),
         ListItem(Paragraph("high（records affected by highest risk，最高风险影响的记录数）是达到最高风险的记录占总体的比例，在Rc较低的情况下high越小，表示风险越低。", styles['Song2']),
                  leftIndent=48, value='circle', bulletColor='#433849', bulletFontSize=6, bulletOffsetY=-3),
         ListItem(Paragraph("uni（sample uniques，唯一样本）是等价组大小为1的记录数量占总体的比例，uni越小，表示风险越低。", styles['Song2']),
                  leftIndent=48, value='circle', bulletColor='#433849', bulletFontSize=6, bulletOffsetY=-3),
         ListItem(Paragraph("priv（private attribute inference attack accuracy，隐私属性推断攻击准确率）模拟攻击者的角色，基于隐私处理后的数据集训练隐私属性预测的机器学习模型，计算隐私属性推断的准确率。priv越低，表示风险越低。", styles['Song2']),
                  leftIndent=48, value='circle', bulletColor='#433849', bulletFontSize=6, bulletOffsetY=-3),
         ], bulletType='bullet'))

    story.append(Paragraph("\0", styles['Song']))
    story.append(Paragraph("3、权衡指标：", styles['Song']))
    story.append(ListFlowable([ListItem(Paragraph("tradeoff（trade-off between utility and risk，风险与效用权衡指标）定义为："
                                                  "（1）对于数据匿名算法：效用指标GIL和风险指标Rb的乘积；" 
                                                  "（2）对于指定效用属性的其他算法：隐私指标priv乘以效用指标acc下降的值；" 
                                                  "（3）对于未指定效用属性的其他算法：隐私指标priv乘以效用指标dist。", styles['Song2']),
                  leftIndent=48, value='circle', bulletColor='#433849', bulletFontSize=6, bulletOffsetY=-3),
                 ListItem(Paragraph("tradeoff越小，表示风险越低，同时效用损失越小。我们基于tradeoff指标自动为用户推荐合适的参数取值。", styles['Song2']),
                  leftIndent=48, value='circle', bulletColor='#433849', bulletFontSize=6, bulletOffsetY=-3),
                 ], bulletType='bullet'))

    doc.build(story)

