from django.shortcuts import render, redirect
from myApp.models import User
from .utils.error import *
import hashlib
from .utils import getHomeData, getChangePasswordData
from .utils import getSelfInfo
from .utils import getTableData
from .utils import getHisotryData
from .utils import getSalaryCharData
from .utils import getCompanyCharData
from .utils import getEducationalCharData
from .utils import getCompanyStatusCharData
from .utils import getAddressCharData
from django.core.paginator import Paginator
from . import word_cloud_picture
from .utils.error import *
import random


# Create your views here.
def login(request):
    if request.method == 'GET':
        return render(request, 'login.html')
    else:
        uname = request.POST.get('username')
        pwd = request.POST.get('password')
        md5 = hashlib.md5()
        md5.update(pwd.encode())
        pwd = md5.hexdigest()
        try:
            user = User.objects.get(username=uname, password=pwd)
            request.session['username'] = user.username
            return redirect('/myApp/home')
        except:
            return errorResponse(request, '用户名或密码出错')


def register(request):
    if request.method == 'GET':
        return render(request, 'register.html')
    else:
        uname = request.POST.get('username')
        pwd = request.POST.get('password')
        checkpwd = request.POST.get('checkPassword')

        try:
            User.objects.get(username=uname)
        except:
            if not uname or not pwd or not checkpwd: return errorResponse(request, '不允许为空！')
            if pwd != checkpwd: return errorResponse(request, '两次密码不符合！')
            md5 = hashlib.md5()
            md5.update(pwd.encode())
            pwd = md5.hexdigest()
            User.objects.create(username=uname, password=pwd)
            return redirect('/myApp/login')

        return errorResponse(request, '该用户名已经被注册！')


def logOut(request):
    request.session.clear()
    return redirect('login', )


def home(request):
    uname = request.session.get('username')
    userInfo = User.objects.get(username=uname)
    # 怕之后的.utils中的辅助方法太多会导致方法名字相同
    yea, month, day = getHomeData.getNowTime()
    userCreateData = getHomeData.getUserCreateTime()
    top6Users = getHomeData.getUserTop6()
    jobsLen, usersLen, educationsTop, salaryTop, addressTop, salaryMonthTop, praticeMax = getHomeData.getAllTags()
    jobsPBarData = getHomeData.getAllJobsPBar()
    tablaData = getHomeData.getTablaData()
    return render(request, 'home.html', {
        'userInfo': userInfo,
        'dataInfo': {
            'year': yea,
            'month': month,
            'day': day
        },
        'userCreateData': userCreateData,
        'top6Users': top6Users,
        'tagDic': {
            'jobsLen': jobsLen,
            'usersLen': usersLen,
            'educationsTop': educationsTop,
            'salaryTop': salaryTop,
            'addressTop': addressTop,
            'salaryMonthTop': salaryMonthTop,
            "praticeMax": praticeMax
        },
        'jobsPBarData': jobsPBarData,
        'tableData': tablaData,
    })


def selfInfo(request):
    uname = request.session.get('username')
    userInfo = User.objects.get(username=uname)
    educations, workExperience, jobList = getSelfInfo.getPageData()
    if request.method == 'POST':
        getSelfInfo.changeSelfInfo(request.POST, request.FILES)
        userInfo = User.objects.get(username=uname)
    return render(request, 'selfInfo.html', {
        'userInfo': userInfo,
        'pageData': {
            'educations': educations,
            'workExperience': workExperience,
            'jobList': jobList
        }
    })


def changePassword(request):
    uname = request.session.get('username')
    userInfo = User.objects.get(username=uname)
    if request.method == 'POST':
        res = getChangePasswordData.changePassword(userInfo, request.POST)
        if res != None:
            return errorResponse(request, res)
        userInfo = User.objects.get(username=uname)
    return render(request, 'changePassword.html', {
        'userInfo': userInfo
    })


def tableData(request):
    uname = request.session.get('username')
    userInfo = User.objects.get(username=uname)
    tableData = getTableData.getTableData()
    paginator = Paginator(tableData, 10)
    cur_page = 1
    if request.GET.get('page'): cur_page = int(request.GET.get('page'))
    c_page = paginator.page(cur_page)

    page_range = []
    visibleNumber = 10
    min = int(cur_page - visibleNumber / 10)
    if min < 1:
        min = 1
    max = min + visibleNumber
    if max > paginator.page_range[-1]:
        max = paginator.page_range[-1]
    for i in range(min, max):
        page_range.append(i)

    return render(request, 'tableData.html', {
        'userInfo': userInfo,
        'c_page': c_page,
        'page_range': page_range,
        'paginator': paginator
    })


def historyTableData(request):
    uname = request.session.get('username')
    userInfo = User.objects.get(username=uname)
    historyData = getHisotryData.getHisotryData(userInfo)
    return render(request, 'historyTableData.html', {
        'userInfo': userInfo,
        'historyData': historyData
    })


def addHistory(request, jobId):
    uname = request.session.get('username')
    userInfo = User.objects.get(username=uname)
    getHisotryData.addHistory(userInfo, jobId)
    return redirect('historyTableData')


def removeHisotry(request, hisId):
    getHisotryData.removeHisotry(hisId)
    return redirect('historyTableData')


def salary(request):
    uname = request.session.get('username')
    userInfo = User.objects.get(username=uname)
    educations, workExperiences = getSalaryCharData.getPageData()
    defaultEducation = '不限'
    defaultWorkExperience = '不限'
    if request.GET.get('educational'): defaultEducation = request.GET.get('educational')
    if request.GET.get('workExperience'): defaultWorkExperience = request.GET.get('workExperience')
    salaryList, barData, legends = getSalaryCharData.getBarData(defaultEducation, defaultWorkExperience)
    pieData = getSalaryCharData.pieData()
    louDouData = getSalaryCharData.getLouDouData()
    return render(request, 'salaryChar.html', {
        'userInfo': userInfo,
        'educations': educations,
        'workExperiences': workExperiences,
        'defaultEducation': defaultEducation,
        'defaultWorkExperience': defaultWorkExperience,
        'salaryList': salaryList,
        'barData': barData,
        'legends': legends,
        'pieData': pieData,
        'louDouData': louDouData
    })


def company(request):
    uname = request.session.get('username')
    userInfo = User.objects.get(username=uname)
    typeList = getCompanyCharData.getPageData()
    type = 'all'
    if request.GET.get('type'): type = request.GET.get('type')
    rowBarData, columnBarData = getCompanyCharData.getCompanyBar(type)
    pieData = getCompanyCharData.getCompanyPie(type)
    companyPeople, lineData = getCompanyCharData.getCompanPeople(type)
    return render(request, 'companyChar.html', {
        'userInfo': userInfo,
        'typeList': typeList,
        "type": type,
        "rowBarData": rowBarData,
        "columnBarData": columnBarData,
        'pieData': pieData,
        "companyPeople": companyPeople,
        "lineData": lineData
    })


def companTags(request):
    uname = request.session.get('username')
    userInfo = User.objects.get(username=uname)
    return render(request, 'companyTags.html', {
        'userInfo': userInfo
    })


def educational(request):
    uname = request.session.get('username')
    userInfo = User.objects.get(username=uname)
    defaultEducation = '不限'
    if request.GET.get('educational'): defaultEducation = request.GET.get('educational')
    educations = getEducationalCharData.getPageData()
    workExperiences, charDataColumnOne, charDataColumnTwo, hasEmpty = getEducationalCharData.getExpirenceData(
        defaultEducation)
    barDataRow, barDataColumn = getEducationalCharData.getPeopleData()
    return render(request, 'educationalChar.html', {
        'userInfo': userInfo,
        'educations': educations,
        'defaultEducation': defaultEducation,
        'workExperiences': workExperiences,
        'charDataColumnOne': charDataColumnOne,
        'charDataColumnTwo': charDataColumnTwo,
        'hasEmpty': hasEmpty,
        'barDataRow': barDataRow,
        'barDataColumn': barDataColumn
    })


def companyStatus(request):
    uname = request.session.get('username')
    userInfo = User.objects.get(username=uname)
    defaultType = '不限'
    if request.GET.get('type'): defaultType = request.GET.get('type')
    typeList = getCompanyStatusCharData.getPageData()
    teachnologyRow, teachnologyColumn = getCompanyStatusCharData.getTechnologyData(defaultType)
    companyStatusData = getCompanyStatusCharData.getCompanyStatusData()
    return render(request, 'companyStatusChar.html', {
        'userInfo': userInfo,
        'typeList': typeList,
        'defaultType': defaultType,
        'teachnologyRow': teachnologyRow,
        'teachnologyColumn': teachnologyColumn,
        'companyStatusData': companyStatusData
    })


def address(request):
    uname = request.session.get('username')
    userInfo = User.objects.get(username=uname)
    defaultCity = '北京'
    if request.GET.get('city'): defaultCity = request.GET.get('city')
    hotCities = getAddressCharData.getPageData()
    salaryRows, salaryColumns = getAddressCharData.getSalaryData(defaultCity)
    companyPeopleData = getAddressCharData.companyPeopleData(defaultCity)
    educationData = getAddressCharData.getEducationData(defaultCity)
    distData = getAddressCharData.getDistData(defaultCity)
    randomPicture = random.randint(1, 1000000)
    word_cloud_picture.get_img('companyTags', './static/3.png', './static/' + str(randomPicture) + '.png')
    return render(request, 'addressChar.html', {
        'userInfo': userInfo,
        'hotCities': hotCities,
        'defaultCity': defaultCity,
        'salaryRows': salaryRows,
        'salaryColumns': salaryColumns,
        'companyPeopleData': companyPeopleData,
        'educationData': educationData,
        'distData': distData,
        'url': randomPicture
    })


import joblib
import pandas as pd
import re

# 加载模型和编码器（只加载一次也行，可以提到全局）
encoder = joblib.load('myApp/models/salary_encoder.pkl')
rf_model = joblib.load('myApp/models/salary_rf_model.pkl')
xgb_model = joblib.load('myApp/models/salary_xgb_model.pkl')


def preprocess_input(input_data):
    """预处理输入数据"""
    df = pd.DataFrame([input_data])

    def get_company_size(x):
        nums = re.findall(r'\d+', str(x))
        return max(map(int, nums)) if nums else 100

    df['company_size'] = df['companyPeople'].apply(get_company_size)

    df['is_listed'] = df['companyStatus'].apply(lambda x: 1 if '不需要融资' in str(x) else 0)

    edu_map = {'学历不限': 0, '大专': 1, '本科': 2, '硕士': 3, '博士': 4}
    df['education_code'] = df['educational'].map(edu_map).fillna(0)

    def get_experience(x):
        x = str(x)
        if '1-3年' in x:
            return 1
        elif '3-5年' in x:
            return 2
        elif '5-10年' in x:
            return 3
        elif '10年以上' in x:
            return 4
        else:
            return 0

    df['experience_code'] = df['workExperience'].apply(get_experience)

    encoded = encoder.transform(df[['address', 'type']])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['address', 'type']))

    features = pd.concat([
        df[['experience_code', 'education_code', 'company_size', 'is_listed']],
        encoded_df
    ], axis=1)

    return features


def salarypredict(request):
    uname = request.session.get('username')
    userInfo = User.objects.get(username=uname)

    predicted_salary = None

    if request.method == 'GET':
        modelType = request.GET.get('modelType')
        address = request.GET.get('address')
        type = request.GET.get('type')
        educational = request.GET.get('educational')
        workExperience = request.GET.get('workExperience')
        companyPeople = request.GET.get('companyPeople')
        companyStatus = request.GET.get('companyStatus')

        # 进行数据检查，确保所有必需字段都存在并且有效
        if not (address and type and educational and workExperience and companyPeople and companyStatus):
            predicted_salary = ""
        else:
            # 准备输入数据
            input_data = {
                'address': address,
                'type': type,
                'educational': educational,
                'workExperience': workExperience,
                'companyPeople': companyPeople,
                'companyStatus': companyStatus
            }

            # 预处理输入数据
            X = preprocess_input(input_data)

            # 根据选择的模型进行预测
            if modelType == 'random_forest':
                predicted_salary = rf_model.predict(X)[0]
            elif modelType == 'gradient_boosting':
                predicted_salary = xgb_model.predict(X)[0]
            else:
                predicted_salary = "无效的模型类型"

            # 保证预测结果是数字类型并格式化为浮动的两位小数
            predicted_salary = round(float(predicted_salary), 2)
            # print(predicted_salary)

    return render(request, 'salaryPredict.html', {
        'userInfo': userInfo,
        'predicted_salary': predicted_salary,
    })


import json
import os
from io import BytesIO
from django.http import HttpResponse
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

def exportReport(request):

    # 获取前端表单数据
    model_type_raw = request.GET.get('modelType', '')
    model_type_map = {
        'random_forest': '随机森林模型',
        'gradient_boosting': '梯度提升模型'
    }
    model_type_display = f"{model_type_map.get(model_type_raw, '未知模型')}（{model_type_raw}）"

    # 处理公司规模字段，转成 x~X人 格式
    company_people_raw = request.GET.get('companyPeople', '')

    def format_company_people(cp_str):
        try:
            # 尝试把字符串转成列表
            cp_list = json.loads(cp_str.replace("'", '"'))
            if isinstance(cp_list, list) and len(cp_list) == 2:
                return f"{cp_list[0]}至{cp_list[1]}人"
            else:
                return cp_str  # 非预期格式，原样返回
        except Exception:
            # 如果不是列表格式，检查是否已经是区间字符串
            if '~' in cp_str:
                return f"{cp_str}人"
            else:
                return cp_str

    company_people_display = format_company_people(company_people_raw)


    # 获取所有表单字段
    input_data = {
        '使用预测的模型': model_type_display,
        '工作城市': request.GET.get('address', '未选择'),
        '职位类型': request.GET.get('type', '未选择'),
        '学历要求': request.GET.get('educational', '未选择'),
        '工作经验': request.GET.get('workExperience', '未选择'),
        '公司规模': company_people_display,
        '融资状态': request.GET.get('companyStatus', '未选择'),
    }

    # 获取预测薪资
    predicted_salary = request.GET.get('predicted_salary', '无预测结果')

    # 注册中文字体：宋体
    font_path = os.path.join('myApp/fonts/simsun.ttc')
    pdfmetrics.registerFont(TTFont('SimSun', font_path))

    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 80

    # 大标题
    p.setFont("SimSun", 20)
    p.drawString(100, y, "Python 工程师薪资预测报告")

    # 横线
    y -= 20
    p.setFont("SimSun", 14)
    p.drawString(100, y, "————————————————————————————————————————————")

    # 一、输入信息
    y -= 40
    p.setFont("SimSun", 16)
    p.drawString(100, y, "一、输入信息")

    y -= 30
    p.setFont("SimSun", 14)
    for key, value in input_data.items():
        p.drawString(100, y, f"{key}：{value}")
        y -= 24
        if y < 100:
            p.showPage()
            y = height - 80
            p.setFont("SimSun", 14)

    # 空行
    y -= 10

    # 二、预测薪资
    p.setFont("SimSun", 16)
    p.drawString(100, y, "二、预测的薪资")

    y -= 30
    p.setFont("SimSun", 14)
    p.drawString(100, y, f"{predicted_salary} 元 / 月")

    # 横线
    y -= 40
    p.drawString(100, y, "————————————————————————————————————————————")

    # 结尾说明
    y -= 40
    p.setFont("SimSun", 13)
    p.drawString(100, y, "本报告为AI生成，结果仅供参考")

    # 保存并返回
    p.showPage()
    p.save()
    buffer.seek(0)
    return HttpResponse(buffer, content_type='application/pdf')


def moreAction(request):
    uname = request.session.get('username')
    userInfo = User.objects.get(username=uname)
    return render(request, 'MoreAction.html', {
        'userInfo': userInfo,
    })

# import joblib
# import numpy as np
# from django.shortcuts import render
# from .models import User


def morePredict(request):
    uname = request.session.get('username')
    userInfo = User.objects.get(username=uname)
    prediction_result = None

    if request.method == 'POST':
        model_type = request.POST.get('modelType', 'random_forest')
        model_path = 'myApp/models/salary_predictor01.pkl' if model_type == 'random_forest' else 'myApp/models/salary_predictor02.pkl'

        model = joblib.load(model_path)

        # 构造输入特征
        input_data = {
            'educational': int(request.POST.get('educational')),
            'workExperience': float(request.POST.get('workExperience')),
            'salaryMonth': float(request.POST.get('salaryMonth')),
            'companyPeople': float(request.POST.get('companyPeople')),
            'workTag_count': int(request.POST.get('workTag_count')),
            'companyTags_count': int(request.POST.get('companyTags_count')),
            'address': request.POST.get('address'),
            'type': request.POST.get('type'),
            'companyNature': request.POST.get('companyNature')
        }

        import pandas as pd
        df = pd.DataFrame([input_data])
        prediction = model.predict(df)[0]
        prediction_result = round(prediction, 2)

    return render(request, 'morePredict.html', {
        'userInfo': userInfo,
        'prediction_result': prediction_result
    })

