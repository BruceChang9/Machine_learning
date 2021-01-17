#20200725
#第五章 if语句
cars=['bmw','audi','benz']
for i in cars:
    if i =='audi':
        print(i.upper())
    else:
        print(i.title())
        
cars=['bmw','audi','benz']
for i in cars:
    if i !='audi':#检查是否为不等于
        print(i.upper())
    else:
        print(i.title())

#if-elif-else为判断是否满足一个条件；
#连续的if可以一直判断是否满足多个条件。
seasons=['spring','summer','fall','winter']
if 'spring' in seasons:
    print('真是太美了')
elif 'summer' in seasons:
    print('真是太热了')
elif 'fall' in seasons:
    print('真是太凄美了')
else:
    print('真是太冷了')

seasons=['spring','summer','fall','winter']
if 'spring' in seasons:
    print('真是太美了')
if 'summer' in seasons:
    print('真是太热了')
if 'fall' in seasons:
    print('真是太凄美了')
if 'winter' in seasons:
    print('真是太冷了')

#有时也可以不用else而是使用if-elif语句
    
seasons=['spring','summer','fall','winter']
if seasons:#检查是否为空列表
    for i in seasons:
        print('这是'+i.title()+'\n')
    print("到此为止")
else:
    print('你是认真的吗？')
    
my_seasons=['spring','summer']
seasons=['spring','summer','fall','winter']
for i in seasons:
    if i in my_seasons:
        print('I love'+i.title()+'so much!')
    else:
        print(i.upper()+'是个啥子东西')
print('到此为止')