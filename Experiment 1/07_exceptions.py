def divide(a, b):
    division_result = None
    try:
        quotient = a / b
    except ZeroDivisionError:
        error_message = "Cannot divide by zero"
        print(error_message)
    else:
        success_message = "Division successful"
        print(success_message)
        division_result = quotient
    finally:
        done_message = "Done trying division"
        print(done_message)
    return division_result

numerator = 10
good_divisor = 2
bad_divisor = 0

first_result = divide(numerator, good_divisor)
print(first_result)

second_result = divide(numerator, bad_divisor)
print(second_result)

class InvalidAgeError(Exception):
    pass

def check_age(age):
    if age < 0:
        negative_age_message = "Age cannot be negative"
        raise InvalidAgeError(negative_age_message)
    return age

invalid_age = -5

try:
    checked_age = check_age(invalid_age)
except InvalidAgeError as e:
    print(e)
