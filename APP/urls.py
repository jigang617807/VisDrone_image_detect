# from django.conf.urls import url
from django.urls import path
from App import views
app_name = 'app'
urlpatterns= [
    path(r'index/',views.index,name='index'),
    path(r'register/', views.register, name='register'),
    path(r'stulogin/', views.stulogin, name='stulogin'),
    path(r'imagefield/', views.image_field, name='image_field'),  # 上传文件的简单写法，写入数据库
    path(r'mine/', views.mine, name='mine'),  # 获取用户信息，例如写出图像或者写出用户名等等
    path(r'getcode/',views.get_code,name='get_code'),#验证码
    path(r'stumine/',views.stumine,name='stumine'), #用户主页面
    path(r'detect/',views.detect,name='detect'), #检测图像
    path(r'dele_pic/',views.dele_pic,name='dele_pic'), #不彻底删除用户的一张图像，会被放入回收站
    path(r'dele/',views.dele,name='dele'), #彻底！删除用户的一张图像
    path(r'rubbish/',views.rubbish,name='rubbish'), #彻底！删除用户的一张图像
    #path(r'map/',views.map,name='map'),
    path(r'work/',views.work,name='work'),
    path(r'hitory/',views.hitory,name='hitory'),
    path(r'recycle/',views.recycle,name='recycle'),
]