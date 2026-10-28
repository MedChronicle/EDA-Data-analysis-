fruits = ["apple", "banana", "cherry"]

new_fruit = "date"
insert_fruit = "avocado"
insert_index = 1

fruits.append(new_fruit)
fruits.insert(insert_index, insert_fruit)

slice_start = 1
slice_stop = 3
fruit_slice = fruits[slice_start:slice_stop]
reversed_fruits = fruits[::-1]

print(fruits)
print(fruit_slice)
print(reversed_fruits)

coordinates = (10.0, 20.0)
x, y = coordinates

print(x, y)

a_set = {1, 2, 3, 4}
b_set = {3, 4, 5, 6}

union_set = a_set | b_set
intersection_set = a_set & b_set
difference_set = a_set - b_set

print(union_set)
print(intersection_set)
print(difference_set)

student = {"name": "Bob", "age": 22, "major": "CS"}

gpa_value = 3.8
student["gpa"] = gpa_value

for key, value in student.items():
    print(key, value)

range_start = 1
range_stop = 11
squares = [n ** 2 for n in range(range_start, range_stop)]

print(squares)

evens_limit = 20
evens_only = [n for n in range(evens_limit) if n % 2 == 0]

print(evens_only)

word_lengths = {word: len(word) for word in fruits}

print(word_lengths)
