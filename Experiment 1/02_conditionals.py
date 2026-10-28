def classify_number(n):
    if n > 0:
        label = "positive"
    elif n < 0:
        label = "negative"
    else:
        label = "zero"
    return label

numbers = [10, -5, 0]

for num in numbers:
    classification = classify_number(num)
    print(num, classification)

start = 1
stop = 16

for i in range(start, stop):
    if i % 15 == 0:
        output = "FizzBuzz"
    elif i % 3 == 0:
        output = "Fizz"
    elif i % 5 == 0:
        output = "Buzz"
    else:
        output = i
    print(output)

count = 0
limit = 10
max_print = 7

while count < limit:
    count += 1
    is_even = count % 2 == 0
    if is_even:
        continue
    if count > max_print:
        break
    print(count)
