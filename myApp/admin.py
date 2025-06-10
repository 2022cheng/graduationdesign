from django.contrib import admin

# Register your models here.
from django.contrib import admin
from myApp.models import JobInfo,User,History

class JobManager(admin.ModelAdmin):
    list_display = ["id","title", "address", "type", "educational", "workExperience", "workTag", "salary", "salaryMonth",
                     "companyTags", "hrWork", "hrName", "pratice", "companyTitle", "companyAvatar", "companyNature",
                     "companyStatus", "companyPeople", "detailUrl", "companyUrl", "dist"]
    list_display_links = ["title"]
    list_editable = ["address", "type", "educational", "workExperience", "workTag", "salary", "salaryMonth",
                     "companyTags", "hrWork", "hrName", "pratice", "companyTitle", "companyAvatar", "companyNature",
                     "companyStatus", "companyPeople", "detailUrl", "companyUrl", "dist"]
    list_filter = ['type']
    search_fields = ['title']
    readonly_fields = ['id']
    list_per_page = 20
    date_hierarchy = 'createTime'

class UserManager(admin.ModelAdmin):
    list_display = ["id",'username','password','avatar','address','educational','work','workExpirence']
    list_display_links = ["username"]
    list_editable = ['password','avatar','address','educational','work','workExpirence']
    search_fields = ['username']
    readonly_fields = ['id']
    list_per_page = 20
    date_hierarchy = 'createTime'

class HistoryManager(admin.ModelAdmin):
    list_display = ["id", "job","user","count"]

admin.site.register(JobInfo,JobManager)
admin.site.register(User,UserManager)
admin.site.register(History,HistoryManager)
