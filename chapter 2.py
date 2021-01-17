#20200723
#第二章 变量和简单数据类型
name='ada lovelace'
print(name.title())
#title()以首字母大写的方式显示每一个单词
print(name.upper())#全部大写
print(name.lower())#全部小写

first_name='chang'
last_name='huaiwen'
print('Hello,'+first_name.title()+last_name.title()+'!')
print('\tPython')#缩进字符\t
print('\tPython\n\tI love you')#换行\n

favorite_language=' Python '
favorite_language.rstrip()#右删空格
favorite_language.lstrip()#左删空格
favorite_language.strip()#删空格

a=2**3#乘方
age=23
age_new=str(age)#将整数int变量转换为字符串str变量

