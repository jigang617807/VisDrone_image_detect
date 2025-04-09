# VisDrone_image_detect
一个无人机图像小目标检测项目，使用yolov5+SwinTransformer+Django配合实现。

底层逻辑是构建数据表Mysql在Settings.py配置自己的数据库，然后数据迁移  python manange.py makemigration  python manage.py migrate

实际演示：
https://github.com/user-attachments/assets/2d79513b-cef1-4794-942a-b3b39e338efb

**复现**
首先是mysql的配置，在settings.py文件里面，有过Django基础的会好一些。
在models.py文件里面是定义了一些数据表的属性以及1:1,1:M等属性设置。还有对应图的存放路径
在views.py文件里面主要写的是底层逻辑。这里主播有点偷懒，对文件路径拆分直接是数数字来的，比如存放是E:/无人机图像小目标检测/data/image/xxxx.jpg.
我直接就数了前面的无效数据（路径）的长度然后做了处理，可以调用python库函数去处理的。这里写死了不容易复现。
