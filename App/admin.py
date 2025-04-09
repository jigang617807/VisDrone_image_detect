from django.contrib import admin

# Register your models here.
from App.models import Student, UserModel, Recycle_Bin

admin.site.register([Student, UserModel, Recycle_Bin])
