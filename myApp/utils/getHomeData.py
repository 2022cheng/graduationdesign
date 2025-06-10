from .getPublicData import *
import time
from datetime import datetime
import json


def getNowTime():
    timeFormat = time.localtime()
    year = timeFormat.tm_year
    mon = timeFormat.tm_mon
    day = timeFormat.tm_mday
    return year, monthList[mon - 1], day


# def getUserCreateTime():
#     users = getAllUsers()
#     data = {}
#     for u in users:
#         if data.get(str(u.createTime),-1) == -1:
#             data[str(u.createTime)] = 1
#         else:
#             data[str(u.createTime)] += 1
#     result = []
#     for k,v in data.items():
#         result.append({
#             'name':k,
#             'value':v
#         })
#     return result
def getUserCreateTime():
    users = getAllUsers()
    data = {}
    for u in users:
        create_date = str(u.createTime)[:10]  # 只取前10位，即"YYYY-MM-DD"
        if data.get(create_date, -1) == -1:
            data[create_date] = 1
        else:
            data[create_date] += 1
    result = []
    for k, v in data.items():
        result.append({
            'name': k,
            'value': v
        })
    return result


def getUserTop6():
    users = getAllUsers()

    def sort_fn(item):
        # 提取日期部分（去掉时分秒）
        date_str = str(item.createTime).split(" ")[0]  # 只取 'YYYY-MM-DD'
        dt = datetime.strptime(date_str, '%Y-%m-%d')  # 解析年月日
        return time.mktime(dt.timetuple())  # 转换为时间戳用于排序

    users = list(sorted(users, key=sort_fn, reverse=True))[:6]
    return users


def getAllTags():
    jobs = JobInfo.objects.all()
    users = User.objects.all()
    educationsTop = '学历不限'
    salaryTop = 0
    salaryMonthTop = 0
    address = {}
    pratice = {}
    for job in jobs:
        if educations[job.educational] < educations[educationsTop]:
            educationsTop = job.educational
        if job.pratice == 0:
            salary = json.loads(job.salary)[1]
            if salaryTop < salary:
                salaryTop = salary
        if int(job.salaryMonth) > salaryMonthTop:
            salaryMonthTop = int(job.salaryMonth)
        if address.get(job.address, -1) == -1:
            address[job.address] = 1
        else:
            address[job.address] += 1
        if pratice.get(job.pratice, -1) == -1:
            pratice[job.pratice] = 1
        else:
            pratice[job.pratice] += 1

    addressStr = sorted(address.items(), key=lambda x: x[1], reverse=True)[:3]
    addressTop = ''
    praticeMax = sorted(pratice.items(), key=lambda x: x[1], reverse=True)
    for index, item in enumerate(addressStr):
        if index == len(addressStr) - 1:
            addressTop += item[0]
        else:
            addressTop += item[0] + ','
    return len(jobs), len(users), educationsTop, salaryTop, addressTop, salaryMonthTop, praticeMax[0][0]


def getAllJobsPBar():
    jobs = getAllJobs()
    tempData = {}

    for job in jobs:
        date_str = str(job.createTime).split()[0]  # 只保留 YYYY-MM-DD
        if tempData.get(date_str, -1) == -1:
            tempData[date_str] = 1
        else:
            tempData[date_str] += 1

    def sort_fn(item):
        return time.mktime(time.strptime(item[0], '%Y-%m-%d'))  # 修正解析错误

    result = list(sorted(tempData.items(), key=sort_fn, reverse=False))

    def map_fn(item):
        item = list(item)
        item.append(round(item[1] / len(jobs), 3))
        return item

    result = list(map(map_fn, result))
    return result


# def getTablaData():
#     jobs = getAllJobs()
#     for i in jobs:
#         # 解析 workTag
#         i.workTag = '/'.join(json.loads(i.workTag))
#
#         # 解析 companyTags
#         if i.companyTags != "无":
#             company_tags_list = json.loads(i.companyTags)
#             if isinstance(company_tags_list, list):
#                 i.companyTags = "/".join(company_tags_list)
#
#         # 解析 companyPeople
#         if i.companyPeople and i.companyPeople.strip() not in ["", "null", "[]", "['']", "None"]:
#             if i.companyPeople == '[0,10000]':
#                 i.companyPeople = '10000人以上'
#             else:
#                 try:
#                     company_people_list = json.loads(i.companyPeople)
#                     if isinstance(company_people_list, list) and all(x == "" for x in company_people_list):
#                         i.companyPeople = "未知人数"
#                     else:
#                         i.companyPeople = '-'.join(f"{x}人" for x in company_people_list)
#                 except json.JSONDecodeError:
#                     i.companyPeople = "未知人数"  # 防止存储数据不是 JSON 格式
#         else:
#             i.companyPeople = "未知人数"
#
#         # 解析 salary
#         if i.salary and i.salary.strip() != "":
#             salary_list = json.loads(i.salary)
#             if isinstance(salary_list, list) and len(salary_list) > 1:
#                 i.salary = salary_list[1]
#             else:
#                 i.salary = "薪资未知"
#         else:
#             i.salary = "薪资未知"
#
#     return jobs

def getTablaData():
    jobs = getAllJobs()
    for i in jobs:
        i.workTag = '/'.join(json.loads(i.workTag))
        if i.companyTags != "无":
            i.companyTags = "/".join(json.loads(i.companyTags)[0].split('，'))
        if i.companyPeople == '[0,10000]':
            i.companyPeople = '10000人以上'
        else:
            i.companyPeople = json.loads(i.companyPeople)
            i.companyPeople = list(map(lambda x:str(x) + '人',i.companyPeople))
            i.companyPeople = '-'.join(i.companyPeople)
        i.salary = json.loads(i.salary)[1]
    return jobs


