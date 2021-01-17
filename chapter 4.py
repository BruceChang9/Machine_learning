#20200725
#第四章 操作列表
magicians=['alice','david','carolina']
for magician in magicians:
    print("\t"+magician.title()+',that was a great trick!')
    print("I can't wait to see your next trick,"+magician.title()+".\n")

for i in range(1,6):
    print(i)

squares=[]
for i in range(1,11):
    a=i**2
    squares.append(a)
print(squares)

squares_new=[i**2 for i in range(1,6)]#列表解析

list(range(1,6))#将数字转换成列表

digits=[1,2,3,4,5]
min(digits)#求最小值
max(digits)#求最大值
sum(digits)#求和

#列表切片
#初始位置至最后一个元素位置+1
players=['alice','tom','jack','han']
print(players[0:3])#第一个至第二个
print(players[:3])
print(players[1:])
print(players[-2:])#最后两个元素

my_foods=['ice cream','vegetables','cakes']
my_friend_foods=my_foods[-2:]
my_foods.append('cookies')
my_friend_foods.append('burger')
print('my favorite foods are:')
print(my_foods)
print("my friend's favorite foods are:")
print(my_friend_foods)

#元组:相较于列表不可改变
a=(1,2)
print(a)
