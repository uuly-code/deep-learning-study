class Dog:
    def __init__(self,name,age):
        self.name=name
        self.age=age
    def introduce(self):
        print(f"我叫{self.name}、今年{self.age}岁")
    def bark(self):
        print(f"{self.name}:汪汪！")

d=Dog("旺财",3)
d.introduce()
d.bark()