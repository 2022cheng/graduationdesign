import json
from myApp.models import JobInfo

def getTableData():
    jobs = list(JobInfo.objects.all())

    def map_fn(item):
        item.salary = json.loads(item.salary)
        item.companyPeople = json.loads(item.companyPeople)
        item.workTag = json.loads(item.workTag)
        if item.companyTags != '无':
            item.companyTags = json.loads(item.companyTags)[0].split('，')
        if not item.pratice:
            item.salary = list(map(lambda x:str(int(x / 1000)),item.salary))
        else:
            item.salary = list(map(lambda x:str(x),item.salary))
        item.salary = '-'.join(item.salary)
        item.companyPeople = list(map(lambda x:str(x),item.companyPeople))
        item.companyPeople = '-'.join(item.companyPeople)
        return item

    jobs = list(map(map_fn,jobs))
    return jobs

