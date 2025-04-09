# from django.conf.urls import url
from django.urls import path

from yolo import views

app_name = 'yolo'
urlpatterns= [
    path(r'index/',views.index,name='index'),

]