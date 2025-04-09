import hashlib
import os
import shutil
import time
import random
from io import BytesIO
from time import sleep
from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import render
from PIL import Image
from PIL.ImageDraw import ImageDraw
from PIL import ImageFont
from django.core.cache import caches
from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.urls import reverse
from django.views.decorators.cache import cache_page

from App.models import Student,UserModel,Recycle_Bin
from App.utils import generatr_code

from yolo import utils
from yolo import yolo_utils
from yolo.utils.general import check_requirements
from yolo.detect import run, parse_opt, go


def index(request):
    return HttpResponse("OK")


def register(request):
    if request.method == "GET":
        return render(request, 'stu_register11.html')
    elif request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        phone = request.POST.get("phone")
        sex = request.POST.get("sex")
        age = request.POST.get("age")
        receive_code = request.POST.get("verify_code")
        store_code = request.session.get("verify_code")
        if receive_code.lower() != store_code.lower():
            print(receive_code.lower())
            print(store_code.lower())
            print("验证码错误")
            return redirect(reverse('app:register'))
        try:
            student = Student()
            student.s_name = username
            student.s_password = password
            student.s_age = age
            student.s_sex = sex
            student.s_phone = phone
            student.save()

        except Exception as e:
            print(e)
            return redirect(reverse('app:register'))

        return render(request, 'stu_login11.html')


def generate_token(ip, username):
    c_time = time.ctime()
    r = username
    # 生成一个token  MD5不知道是啥东西，encode是转编码成二进制，然后后面.hexdigest()的是获取到 字符串 unicode模式
    return hashlib.new("md5", (ip + c_time + r).encode("utf-8")).hexdigest()


def stulogin(request):
    if request.method == "GET":
        # return render(request, 'stu_login11.html')
        return render(request, 'stu_login11.html')
    elif request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        phone = request.POST.get("phone")
        students = Student.objects.filter(s_name=username).filter(s_password=password).filter(s_phone=phone)
        if students.exists():  # 登陆时才加token！
            student = students.first()
            ip = request.META.get("REMOTE_ADDR")
            token = generate_token(ip, username)
            student.s_token = token
            student.save()
            response = render(request, 'Next.html')
            response.set_cookie("token", token)
            receive_code = request.POST.get("verify_code")
            store_code = request.session.get("verify_code")

            if receive_code.lower() != store_code.lower():
                return redirect(reverse('app:stulogin'))
            return response
        else:
            return render(request, 'Login_fault.html')

    return redirect(reverse("app:stulogin"))



def get_code(request):  # 验证码
    # 初始化画布和画笔
    mode = 'RGB'
    size = (330, 100)
    # color_bg=(122,155,10)  # 背景改成随机的不容易识别
    red = random.randrange(256)
    green = random.randrange(256)
    blue = random.randrange(256)
    color_bg = (red, green, blue)
    image = Image.new(mode=mode, size=size, color=color_bg)

    imagedraw = ImageDraw(image, mode=mode)

    imagefont = ImageFont.truetype(settings.FONT_PATH, 100)

    # verify_code="Rock"  #可以改成自己的语料库!
    verify_code = generatr_code()  # 可以改成自己的语料库!
    # 验证码要起到验证作用需要把他存下来！存到cookie里面爬虫可以找到，不推荐
    # 我们存到服务器里面，下面这么做：
    request.session['verify_code'] = verify_code

    for i in range(4):
        fill = (random.randrange(256), random.randrange(256), random.randrange(256))
        imagedraw.text(xy=(85 * i, 0), text=verify_code[i], font=imagefont, fill=fill)  # 使用画笔绘画
    # 这样画出来还是很清晰，我们需要加干扰点

    # 增加干扰点
    for i in range(9000):
        fill = (random.randrange(256), random.randrange(256), random.randrange(256))
        xy = (random.randrange(351), random.randrange(100))
        imagedraw.point(xy=xy, fill=fill)
    # 把绘制好的画转换成二进制文件并且添加格式限定，然后传送到前端
    # 内存IO流
    fp = BytesIO()
    # 存到内存流
    image.save(fp, "png")  # 不创建流会写在本地，造成大量垃圾
    return HttpResponse(fp.getvalue(), content_type='image/png')  # content_type='image/png'不设置这个就会返回乱码，标识打开数据的形式



def stumine(request):  # 个人主页面
    token = request.COOKIES.get("token")
    try:
        student = Student.objects.get(s_token=token)
    except Exception as e:
        redirect(reverse("app:stulogin"))
    # return render(request, 'mine.html', context=locals())
    return render(request, 'index11.html', context=locals())

# 上传文件简洁写法
def image_field(request):
    if request.method == 'GET':
        return render(request, 'work11.html')
    elif request.method == 'POST':
        token = request.COOKIES.get("token")
        try:
            student = Student.objects.get(s_token=token)
        except Exception as e:
            redirect(reverse("app:stulogin"))
        user = UserModel()
        user.u_name=student.s_name
        icon = request.FILES.get('icon')
        user.u_icon = icon
        user.save()
        # student.s_icon = icon
        # student.save()
        user.u_student.add(student)  # 添加映射关系
        user.save()

        stu = Student.objects.get(pk=student.id)  # 找到当前用户的所有图像
        DieDaiQi = stu.usermodel_set.all()  # 将所有图像返回到一个迭代器
        llist = list(DieDaiQi)
        List = []
        if len(llist)!=0:
            for i in range(len(llist)):
                dict = {
                    'id': i + 1,
                    'u_name': str(llist[i].u_name),
                    'u_date': str(llist[i].u_date),
                    'message': str(llist[i].message),
                }
                List.append(dict)
        #
        return render(request,'work11.html',{'List': List})


def mine(request):  # 展示个人用户图片
    token = request.COOKIES.get("token")
    try:
        student = Student.objects.get(s_token=token)
    except Exception as e:
        redirect(reverse("app:stulogin"))
    # print(student.id)
    stu = Student.objects.get(pk=student.id) # 找到当前用户的所有图像
    imglist = stu.usermodel_set.all()  # 将所有图像返回到一个迭代器
    # for img in imglist:
    #     print('/static/'+str(img.u_icon),img.u_name)
    # print(imglist)
    # print("static/upload" + imglist.u_icon.url)
    # data = {
    #     "username": stu.s_name,
    #     "icon_url": imglist
    # }
    # return render(request, 'show_image.html', context=data)
    return render(request, 'show_image111.html', context=locals())

def rubbish(request):
    token = request.COOKIES.get("token")
    try:
        student = Student.objects.get(s_token=token)
    except Exception as e:
        redirect(reverse("app:stulogin"))
    stu = Student.objects.get(pk=student.id) # 找到当前用户的所有图像
    imglist = stu.recycle_bin_set.all()# 将所有图像返回到一个迭代器
    # imglist = stu.usermodel_set.all()  # 将所有图像返回到一个迭代器
    return render(request, 'rubbish111.html', context=locals())

def detect(request):
    img = request.GET.get('img')  # 获取传输过来的img
    img = 'D:/yoloweb'+str(img)  # 补全路径信息

    save_dir,message = go(img)  # 调用检测算法并且返回存储位置和检测信息

    token = request.COOKIES.get("token")  # 获取当前用户信息
    try:
        student = Student.objects.get(s_token=token)  # 匹配当前用户库中是否有该用户，如果有才可以进行下面的存储
    except Exception as e:
        redirect(reverse("app:stulogin"))
    # 获取这个学生名下的所有图片，这是级联部分的操作
    stu = Student.objects.get(pk=student.id) # 找到当前用户的所有图像
    # 获得的数据返回到一个迭代器对象
    imglist = stu.usermodel_set.all()  # 将所有图像返回到一个迭代器
    for uimg in imglist:
        # print(uimg.u_icon)
        if 'D:/yoloweb/static/upload/'+str(uimg.u_icon) == img:
            uimg.after_uicon =str(save_dir)[10:]+'/'+str(uimg.u_icon)[6:]
            # print(len(massage),type(massage))
            # print(len(str(uimg.u_icon)))
            l = len(str(uimg.u_icon))  # 获取连接长度
            x =45+l  # 前面没用信息的长度加上图片长度
            # print(massage[x:])
            # print(massage[x:len(massage)-2])
            uimg.message=message[x:len(message)-2]
            uimg.save()

    return HttpResponse("检测成功")



# 这里写的是不彻底删除，另放到了回收站！
def dele_pic(request):
    request.method='POST'
    token = request.COOKIES.get("token")
    student = Student.objects.get(s_token=token)
    print(student.s_name)
    pid=request.GET.get("pid")
    # 删除“Jmeter接口”这本书以及关系表中的对应关系
    pic_obj = UserModel.objects.get(id=pid)  # 正常get后面加id才能保证结果是唯一的
    rubbish = Recycle_Bin()
    rubbish.r_name=student.s_name
    rubbish.r_icon=pic_obj.u_icon
    rubbish.r_after_uicon=pic_obj.after_uicon
    rubbish.r_message=pic_obj.message
    rubbish.save()
    rubbish.r_student.add(student)  # 添加映射关系
    rubbish.save()
    pic_obj.delete()
    return HttpResponse("删除成功")

# 这里写的是彻底删除，其实按照逻辑来应该是不彻底删除，应为有回收站着歌东西！
def dele(request):
    request.method = 'POST'
    pid=request.GET.get("pid")
    # 删除这条信息以及关系表中的对应关系
    pic_obj = Recycle_Bin.objects.get(id=pid)  # 正常get后面加id才能保证结果是唯一的
    print(pic_obj.r_message)
    dir="D:/yoloweb/static/upload/"
    os.remove(dir + '{}'.format(pic_obj.r_icon))  # 删除本地文件
    dir1="D:/yoloweb"
    index = len(str(pic_obj.r_after_uicon))-len(str(pic_obj.r_icon))+5
    # 递归删除文件夹
    shutil.rmtree(dir1 + '{}'.format(str(pic_obj.r_after_uicon)[0:index]))
    pic_obj.delete()
    return HttpResponse("删除成功")





def work(request):
    token = request.COOKIES.get("token")
    try:
        student = Student.objects.get(s_token=token)
    except Exception as e:
        redirect(reverse("app:stulogin"))
    # print(student.id)
    stu = Student.objects.get(pk=student.id) # 找到当前用户的所有图像
    DieDaiQi = stu.usermodel_set.all()  # 将所有图像返回到一个迭代器
    # print(type(DieDaiQi))
    llist=list(DieDaiQi)
    List = []
    if len(llist)!=0:
        # print(llist)
        for i in range(len(llist)):
            dict={
                'id':i+1,
                'u_name':str(llist[i].u_name),
                'u_date':str(llist[i].u_date),
                'message': str(llist[i].message),
            }
            List.append(dict)
        # ,{'List': List}
        print(List[0])
    return render(request,'work11.html',{'List': List})


def hitory(request):
    # 确定用户
    token = request.COOKIES.get("token")
    try:
        student = Student.objects.get(s_token=token)
    except Exception as e:
        redirect(reverse("app:stulogin"))

    stu = Student.objects.get(pk=student.id)  # 找到当前用户的所有图像
    DieDaiQi = stu.usermodel_set.all()  # 将所有图像返回到一个迭代器

    llist = list(DieDaiQi)

    List = []
    if len(llist)!=0:
        for i in range(len(llist)):
            dict = {
                'id': i + 1,
                'u_name': str(llist[i].u_name),
                'u_date': str(llist[i].u_date),
                'message': str(llist[i].message),
                'u_icon':str(llist[i].u_icon),
                'after_uicon':str(llist[i].after_uicon),
            }
            List.append(dict)
    return render(request,'history11.html',{'List': List})


def recycle(request):
    return render(request,'rubbish111.html')