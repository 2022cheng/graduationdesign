from myApp.models import *

monthList = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
             'November', 'December']
educations = {'博士': 1, '硕士': 2, '本科': 3, '大专': 4, '高中': 5, '中专/中技': 6, '初中及以下': 7, '学历不限': 8}
workExperience = ['在校/应届生', '经验不限', '1-3年', '3-5年', '5-10年', '10年以上']
salaryList = ['0-10k', '10-20k', '20-30k', '30-40k', '40k以上']
companyPeople = ['20人以下', '100人以下', '500人以下', '1000人以下', '1万人以下', '1万人以上']
hotCity = ['北京', '上海', '深圳', '成都', '昆明', "郑州", '重庆', '广州', '东莞', '天津']


def getAllUsers():
    return User.objects.all()


def getAllJobs():
    return JobInfo.objects.all()
