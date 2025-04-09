# 上面生成的验证码不是随机的，是我们自己制定的，下面定义一个函数来获取随机串
import random
def generatr_code():
    source='78915463ASXZEGSLPkl105454dasdwauhuihjSDASXAIUOEJAJNSIDJADASCVKLPOPOSDnaixniauhiufias12R4'
    code = ""
    for i in range(4):
        code+=random.choice(source)
    return code