import math
import random
import datetime

square_root_input = 64
floor_input = 4.7
ceil_input = 4.2

square_root_result = math.sqrt(square_root_input)
pi_value = math.pi
floor_result = math.floor(floor_input)
ceil_result = math.ceil(ceil_input)

print(square_root_result)
print(pi_value)
print(floor_result)
print(ceil_result)

random_min = 1
random_max = 10
numbers = [1, 2, 3, 4, 5]

random_integer = random.randint(random_min, random_max)
random_choice = random.choice(numbers)

print(random_integer)
print(random_choice)

random.shuffle(numbers)

print(numbers)

today = datetime.date.today()
today_year = today.year
today_month = today.month
today_day = today.day

print(today)
print(today_year, today_month, today_day)
