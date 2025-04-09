$(function (){
    z = document.getElementsByClassName("code")
        for (i=0;i<z.length;i++){
        document.getElementsByClassName("code")[i].addEventListener("click", Change);
        console.log(z[i])
    }
    function Change(){
            $(this).attr("src","/app/getcode/?t=" + Math.random())
    }
})