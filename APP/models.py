from django.db import models
# Create your models here.
class Student(models.Model):
    s_name = models.CharField(max_length=16)
    s_password = models.CharField(max_length=128)
    s_sex = models.CharField(max_length=2, default='男')
    s_age = models.IntegerField(default=15)
    s_phone = models.CharField(max_length=20, default='123')
    s_token = models.CharField(max_length=256)
    # s_icon = models.ImageField(upload_to='icons',default=None)


class UserModel(models.Model):
    u_name = models.CharField(max_length=16)
    u_student = models.ManyToManyField(Student)
    # upload_to写的是相对路径，相对于MEDIA_ROOT的根目录
    u_icon = models.ImageField(upload_to='icons',default=None)
    after_uicon = models.ImageField(default=None)
    message = models.TextField(default=None,null=True)
    u_date = models.DateTimeField('图片传入时间', auto_now=True)



class Recycle_Bin(models.Model):
    r_name = models.CharField(max_length=16)
    r_student = models.ManyToManyField(Student)
    # upload_to写的是相对路径，相对于MEDIA_ROOT的根目录
    r_icon = models.ImageField(default=None)
    r_after_uicon = models.ImageField(default=None)
    r_message = models.TextField(default=None)
    r_date = models.DateTimeField('进入垃圾箱时间', auto_now=True)


