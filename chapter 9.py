#20200920
#第九章 类
class Dog():
    
    def __init__(self,name,age):
        self.name=name
        self.age=age
    
    def sit(self):
        print(self.name.title()+" is now sitting.")
        
    def roll_over(self):
        print(self.name.title()+" rolled over!")

my_dog=Dog('Willie',6)
my_dog.sit()
my_dog.roll_over()

class Car():
    
    def __init__(self,make,model,year):
        self.make=make
        self.model=model
        self.year=year
        
        self.odometer_reading = 0
        
    def read_odometer(self):
        print("This car has "+str(self.odometer_reading)+" miles on it.")
        
    def get_descriptive_name(self):
        long_name=str(self.year)+' '+self.make+' '+self.model
        return long_name.title()
#更新里程数
    def update_odometer(self,mileage):
        if mileage>=self.odometer_reading:
            self.odometer_reading=mileage
        else:
            print("You can't roll back an odometer!")
#增加里程数
    def increment_odometer(self,miles):
        self.odometer_reading += miles
                      
my_new_car=Car('audi','a4',2016)
print(my_new_car.get_descriptive_name())

#修改属性的值
#方法一
my_new_car.odometer_reading=23
my_new_car.read_odometer()
#方法二
my_new_car.update_odometer(10)
my_new_car.read_odometer()

my_new_car.increment_odometer(2)
my_new_car.read_odometer()

#子类
class ElectricCar(Car):
    def __init__(self,make,model,year):
        super().__init__(make,model,year)
        self.battery_size=70
    def describe_battery(self):
        print("This car has a "+str(self.battery_size)+"-KWh battery.")
        
my_tesla=ElectricCar('tesla', 'model s', 2016)
print(my_tesla.get_descriptive_name())
my_tesla.describe_battery()


class Battery():
    def __init__(self,battery_size=70):
        self.battery_size=battery_size
        
    def describe_battery(self):
        print("This car has a "+str(self.battery_size)+"-KWh battery.") 
        
    def get_range(self):
        if self.battery_size == 70:
            range = 240
        elif self.battery_size == 85:
            range = 370
            
        message = "This car can go approximately "+str(range)+" miles on a full charge."
        print(message)
        
class ElectricCar(Car):
    def __init__(self,make,model,year):
        super().__init__(make,model,year)
        
        self.battery=Battery(85)#将ElectricCar和Battery两类联系起来
        
my_tesla=ElectricCar('tesla', 'model s', 2016)
print(my_tesla.get_descriptive_name()) #ElectricCar是Car的子类
my_tesla.battery.describe_battery()        
my_tesla.battery.get_range()