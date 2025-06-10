# import jieba
# from matplotlib import pylab as plt
# from wordcloud import WordCloud
# from PIL import Image
# import numpy as np
# from pymysql import connect
# import json
#
# def get_img(field,targetImageSrc,resImageSrc):
#     con = connect(host="localhost",user='root',password='2022wlccj',database='bossinfo',port=3306,charset='utf8mb4')
#     cursor = con.cursor()
#     sql = f"select {field} from jobinfo"
#     cursor.execute(sql)
#     data = cursor.fetchall()
#     text = ''
#

# for item in data:
#         if field == 'companyTags':
#             # 特殊处理companyTags字段，假设它是JSON格式
#             if item[0] and item[0] != '无':
#                 try:
#                     tags = json.loads(item[0])
#                     if tags:  # 确保tags非空
#                         if isinstance(tags, list):
#                             text += ' '.join(tags)
#                         elif isinstance(tags, str):
#                             text += tags
#                 except json.JSONDecodeError:
#                     text += item[0]
#         else:
#             # 其他字段直接拼接
#             if item[0]:
#                 text += item[0]


#     for i in data:
#         text += i[0]
#         # if i[0] != '无':
#         #     companyTagsArr = json.loads(i[0])[0].split('，')
#         #     # print(companyTagsArr)
#         #     for j in companyTagsArr:
#         #         text += j
#     cursor.close()
#     con.close()
#     data_cut = jieba.cut(text,cut_all=False)
#     stop_words = []
#     with open('stopwords.txt','r',encoding='utf8') as rf:
#         for line in rf:
#             if len(line) > 0:
#                 stop_words.append(line.strip())
#     # print(data_cut,stop_words)
#     data_result = [x for x in data_cut if x not in stop_words]
#     string = ' '.join(data_result)
#     # print(string)
#     # 图片
#     img = Image.open(targetImageSrc)
#     img_arr = np.array(img)
#     wc = WordCloud(
#         background_color='white',
#         mask=img_arr,
#         font_path='STHUPO.TTF'
#     )
#     wc.generate_from_text(string)
#
#     # 绘制图片
#     fig = plt.figure(1)
#     plt.imshow(wc)
#     plt.axis('off')
#
#     plt.savefig(resImageSrc,dpi=800)
#
#
# #
# # get_img('companyTags','../static/2.png','../static/companyTags_cloud.png')
# # get_img('title','../static/2.png','../static/title_cloud.png')


import jieba
from matplotlib import pylab as plt
from wordcloud import WordCloud
from PIL import Image
import numpy as np
from pymysql import connect
import json

def get_img(field,targetImageSrc,resImageSrc):
    con = connect(host="localhost",user='root',password='2022wlccj',database='bossinfo',port=3306,charset='utf8mb4')
    cursor = con.cursor()
    sql = f"select {field} from jobinfo"
    cursor.execute(sql)
    data = cursor.fetchall()
    text = ''
    for i in data:
        if i[0] != '无':
            companyTagsArr = json.loads(i[0])[0].split('，')
            for j in companyTagsArr:
                text += j
    cursor.close()
    con.close()
    data_cut = jieba.cut(text,cut_all=False)
    # stop_words = []
    # with open('stopwords.txt','r',encoding='utf8') as rf:
    #     for line in rf:
    #         if len(line) > 0:
    #             stop_words.append(line.strip())
    # data_result = [x for x in data_cut if x not in stop_words]
    string = ' '.join(data_cut)

    # 图片
    img = Image.open(targetImageSrc)
    img_arr = np.array(img)
    wc = WordCloud(
        background_color='white',
        mask=img_arr,
        font_path='STHUPO.TTF'
    )
    wc.generate_from_text(string)

    # 绘制图片
    fig = plt.figure(1)
    plt.imshow(wc)
    plt.axis('off')

    plt.savefig(resImageSrc,dpi=800)



# get_img('companyTags','../static/1.png','../static/companyTags_cloud.png')
