name = "Alice"
age = 30
height = 5.6
is_student = False

name_type = type(name)
age_type = type(age)
height_type = type(height)
is_student_type = type(is_student)

print(name, name_type)
print(age, age_type)
print(height, height_type)
print(is_student, is_student_type)

age_str = str(age)
height_int = int(height)
pi_value = float("3.14159")

print(age_str)
print(height_int)
print(pi_value)

a = 17
b = 5

addition = a + b
subtraction = a - b
multiplication = a * b
division = a / b
floor_division = a // b
modulus = a % b
power = a ** b

print(addition)
print(subtraction)
print(multiplication)
print(division)
print(floor_division)
print(modulus)
print(power)

is_a_greater = a > b
is_equal = a == b
and_result = a > 10 and b < 10
or_result = a < 10 or b < 10

print(is_a_greater)
print(is_equal)
print(and_result)
print(or_result)
