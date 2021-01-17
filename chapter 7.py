#20200731
#用户输入和while循环
prompt="If you tell us who you are,we can personalize the messages you see."
prompt +="\nWhat is your first name?"
#在存储在prompt中的字符串末尾附加一个字符串

name=input(prompt)
print("\n"+'Hello,'+name+"!")

#输入的是字符串 如要进行运算 需要使用int()

4 % 3
#将两数相除返回余数

prompt="If you tell us who you are,we can personalize the messages you see."
prompt +="\nWhat is your first name?"

active=True
while active:
    message=input(prompt)
    
    if message !='quit':
        print('Hello,'+message+'!')
    else:
        active=False
        
    break
#立即退出while循环，不再运行循环中余下的代码
    
current_number=0
while current_number<24:
    current_number=current_number+1
    if current_number % 2 == 0 :
        continue
    print(current_number)
#要返回到循环开头，并根据条件测试结果决定是否继续执行循环

unconfirmed_users=['jack','mike','hellen']
confirmed_users=[]

#在列表之间移动元素
while unconfirmed_users:
    current_user=unconfirmed_users.pop()
    confirmed_users.append(current_user)
    print('Virifying user:'+current_user.title())

print('\nThe following users have been confirmed:')
for user in confirmed_users:
    print('\t'+user.title())

#删除博阿寒特定值的所有列表元素 
unconfirmed_users=['jack','mike','hellen']
print(unconfirmed_users)

while 'mike' in unconfirmed_users:
    unconfirmed_users.remove('mike')

print(unconfirmed_users)

#使用用户输入来填充字典
responses={}

polling_active=True

while polling_active:
    name=input("\nWhat's your name?")
    response=input("Which food would you like to eat?")

    responses[name]=response

    repeat=input("Would you like another person to respond?(yes/no)")
    if repeat=='no':
        polling_active=False
    
print("\n---Poll Result---")
for name,response in responses.items():
    print(name.title()+" would like to eat "+response+"!")   