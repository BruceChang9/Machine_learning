#20200723
#第三章 修改、添加和删除元素
bicycle=['trek','cannodale','redline']
message='my favorite bicycle is '+bicycle[-1].title()+'!'
print(message)

motorcycles=['honda','yamaha']
motorcycles[0]='ducati'#元素替换
motorcycles.append('suzki')#添加
motorcycles.insert(1,'toyota')#指定位置添加

del motorcycles[2]#删除指定的元素

motorcycles_new=motorcycles.pop()#删除末尾元素,并可以使用最后一个元素

motorcycles_new1=motorcycles.pop(1)#删除指定位置元素,并可以使用该元素
#若删除不使用，则用del；若要使用，则用pop
#方法一
motorcycles.remove('ducati')#将ducati删除 只删除第一个值 循环语句判断是否均删除
#方法二
a=('ducati')
motorcycles.remove(a)

cars=['bmw','audi','toyota']
#对列表进行排列
cars.sort()#永久正向排序
cars.sort(reverse=True)#永久反向排序
b=sorted(cars)
cars.reverse()#列表永久反转
