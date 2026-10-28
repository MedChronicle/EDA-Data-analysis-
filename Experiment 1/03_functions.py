def describe_pet(name, species="dog"):
    message = name + " is a " + species
    print(message)

pet_one = "Rex"
pet_two = "Whiskers"
pet_two_species = "cat"

describe_pet(pet_one)
describe_pet(pet_two, pet_two_species)

def sum_all(*args):
    total = sum(args)
    return total

total_result = sum_all(1, 2, 3, 4)
print(total_result)

def show_info(**kwargs):
    for key, value in kwargs.items():
        print(key, value)

pet_age = 3
pet_color = "black"

show_info(age=pet_age, color=pet_color)

square = lambda x: x ** 2

square_input = 9
square_result = square(square_input)
print(square_result)

def factorial(n):
    if n <= 1:
        result = 1
    else:
        result = n * factorial(n - 1)
    return result

factorial_input = 6
factorial_result = factorial(factorial_input)
print(factorial_result)
