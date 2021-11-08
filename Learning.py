import sys

class Car:
    def __init__(self, speed=0):
        self.speed = speed
        self.odometer = 0
        self.time = 0
    def say_state(self):
        print("I'm going kph!".format(self.speed))
    def accelerate(self):
        self.speed += 5
    def brake(self):
        self.speed -= 5
    def step(self):
        self.odometer += self.speed
        self.time += 1
    def average_speed(self):
        return self.odometer / self.time
if __name__ == '__main__':
    my_car_show_distance = sys.argv[1]
    my_car = Car()
    print("I'm a car!")
    while True:
        action = input("What should I do? [A]ccelerate, [B]rake, "
                 "show [O]dometer, or show average [S]peed?").upper()
        if action not in "ABOS" or len(action) != 1:
            print("I don't know how to do that")
        if my_car_show_distance == "yes":
            print("The car has driven  kilometers")
