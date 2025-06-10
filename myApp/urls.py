from django.urls import path, re_path
from myApp import views

urlpatterns = [
    path('login/', views.login, name='login'),
    path('register/', views.register, name='register'),
    path('home/', views.home, name='home'),
    path('logOut/', views.logOut, name='logOut'),
    path('selfInfo/', views.selfInfo, name='selfInfo'),
    path('changePassword/', views.changePassword, name='changePassword'),
    path('tableData/', views.tableData, name='tableData'),
    path('historyTableData/', views.historyTableData, name='historyTableData'),
    path('addHistorys/<int:jobId>', views.addHistory, name='addHistory'),
    path('removeHisotry/<int:hisId>', views.removeHisotry, name='removeHisotry'),
    path('salary/', views.salary, name='salary'),
    path('company/', views.company, name='company'),
    path('companyTags/', views.companTags, name='companTags'),
    path('educational/',views.educational,name='educational'),
    path('companyStatus/', views.companyStatus, name='companyStatus'),
    path('address/', views.address, name='address'),
    path('salarypredict/', views.salarypredict, name='salarypredict'),
    path('moreAction/', views.moreAction, name='moreAction'),
    path('exportReport/',views.exportReport, name='exportReport'),
    path('morePredict',views.morePredict,name='morePredict'),
]
