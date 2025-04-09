$(function () {
    //删除一条检测记录
    //获取类名为dele的所有标签，这里指向所有图片，返回一个列表对象
    z = document.getElementsByClassName("dele")
    //for循环迭代添加点击事件，点击触发函数DEL
    for (i=0;i<z.length;i++){
        document.getElementsByClassName("dele")[i].addEventListener("click", DEL);
        console.log(z[i])
    }
    //DEL函数
    function DEL() {
        console.log("删除这条记录！");
        var id = $(this).attr('id');//获取当前的id元素，用来定位是哪一张图片
        //将垃圾箱图片的src替换为view.dele函数返回的src当然我们这里没有返回src所以图片会不显示
        //这里弄成图片单纯是为了调用这个函数！！！
        //$(this).attr（“src”）获取当前的src值
        //$(this).attr（“src”, value） 将当前对象的src替换为指定的value，这里的src也可以是id，name等html元素的标签
        $(this).attr("src", "/app/dele/?pid="+id)  //指定向一个目录
        setTimeout(function (){  //延时函数，这里点击后停顿0.5s触发重新定向页面函数
        alert("正在删除中，请稍后！")  //弹出一个提示框
        //重定向页面函数，assign里面可以是一个链接。
        //window.location.protocol返回当前端口
        //window.location.pathname返回当前路径名
        window.location.assign(window.location.protocol+window.location.pathname)
        }, 500);
    }



})