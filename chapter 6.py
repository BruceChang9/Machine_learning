#20200731
#第六章 字典
aline_0={'color':'green','points':5}
print(aline_0['color'])
aline_0['x_position']=0
aline_0['y_position']=3
print(aline_0)

aline_0={}
aline_0['color']='green'
aline_0['points']=5
#创建空字典然后填充属性
aline_0['color']='yellow'
#修改字典中的属性

aline_0={'x_position':5,'speed':'fast'}
print('original positionL:'+str(aline_0['x_position']))

if aline_0['speed']=='fast':
    x_increment=3
elif aline_0['speed']=='medium':
    x_increment=2
else:
    x_increment=1

aline_0['x_position']=aline_0['x_position']+x_increment
print('new position:'+str(aline_0['x_position']))

del aline_0['color']
#删除键值对

favorite_language={
    'jack':'python',
    'mike':'java',
    'ken':'ruby'
    }
for name,language in favorite_language.items():
    print(name.title()+"'s favorite language is "+language.title()+"!")
#访问整体用.items();访问键用.keys();访问值用.values()

alines=[]

for aline_number in range(30):
    new_aline={'color':'green','speed':'slow','points':10}
    alines.append(new_aline)
#在列表中镶嵌字典
for aline in alines[:5]:
    print(aline)
print('......\n')

for aline in alines[3:7]:
    if aline['color']=='green':
        aline['color']='yellow'
        aline['speed']='fast'
        aline['points']=14
for aline in alines[:9]:
    print(aline)
print('......')

favorite_language={
    'mike':['python','matlab'],
    'jack':['c','java'],
    'phil':['ruby']
    }
for name,languages in favorite_language.items():
    print('\n'+name.title()+"'s favorite language is:")
    for language in languages:
        print('\t'+language.title())
        
        