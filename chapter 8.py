#20200803
#第八章 函数
def great_user(username):
    print("Hello,"+username.title()+"!")

great_user('mike')
great_user(username='mike')
#等价表达

def describe_pet(pet_name,animal_type='dog'):
    print("My "+animal_type+"'s name is "+pet_name.title()+".")
describe_pet(pet_name='mike')
#无需重复定义animal_type
describe_pet(pet_name='mike',animal_type='cat')
#也可重新def animal_type

def get_formatted_name(first_name,middle_name,last_name):
    full_name=first_name+" "+middle_name+" "+last_name
    return full_name.title()

musician=get_formatted_name("mike","hooker","lee")
print(musician)

#创建字典
def build_person(first_name,last_name,age):
    person={'first':first_name.title(),'last':last_name.title()}
    if age:
        person['age']=age
        return person
    
musician=build_person('jimi','hendrix',27)
print(musician)

def get_formatted_name(first_name,last_name):
    full_name=first_name+" "+last_name
    return full_name.title()

while True:
    print('\nPlease tell me your real name:')
    print("(enter 'quit' at any time to quit if you want)")
    
    f=input("What's your first name:")
    if f == 'quit':
        break
    l=input("What's your last name:")
    if l == 'quit':
        break

    person=get_formatted_name(f,l)
    print("Hello, "+person)
    
def print_models(unprinted_designs,completed_models):
    while unprinted_designs:
        current_model=unprinted_designs.pop()
        completed_models.append(current_model)
        print('\nPrinting model:'+current_model.title())
    
def show_models(completed_models):
    print("\nThe following models have been printed:")
    for i in completed_models:
        print('\t'+i.title())

unprinted_designs=['dodecahedron','robot pendant','iphone case']
completed_models=[]
print_models(unprinted_designs, completed_models)
show_models(completed_models)

#结合位置实参和任意数量实参(*toppings)
def make_pizza(*toppings):
    for i in toppings:
        print(i.title())
        
make_pizza('mushrooms','green peppers','extra cheese')

#使用任意数量的关键字实参(**user_info)
def build_profile(first_name,last_name,**user_info):
    profile={}
    profile['first name']=first_name.title()
    profile['last name']=last_name.title()
    for key,value in user_info.items():
        profile[key]=value
    return profile

a=build_profile('mike', 'jackson', location='princeton',field='physics')
print(a)

#从chapter_8.py中调用函数build_profile
from chapter_8 import build_profile

#使用as给指定函数命名
from chapter_8 import build_person as bp

#使用as给模块命名
import chapter_8 as c8
c8.build_person()

#导入模块中的所有函数
from chapter_8 import *

