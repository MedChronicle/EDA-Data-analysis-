text = "  Python is Powerful, Python is Elegant!  "

old_word = "Python"
new_word = "Java"

stripped_text = text.strip()
lower_text = text.lower().strip()
upper_text = text.upper().strip()
replaced_text = text.replace(old_word, new_word).strip()
split_text = text.split()
python_count = text.count(old_word)

print(stripped_text)
print(lower_text)
print(upper_text)
print(replaced_text)
print(split_text)
print(python_count)

words = ["Data", "Science", "with", "Python"]

space_separator = " "
dash_separator = "-"

space_joined = space_separator.join(words)
dash_joined = dash_separator.join(words)

print(space_joined)
print(dash_joined)

sample = "Hello, World!"

first_five = sample[:5]
last_six = sample[-6:]
reversed_sample = sample[::-1]

print(first_five)
print(last_six)
print(reversed_sample)

name = "Sara"
score = 92.5

fstring_message = f"{name} scored {score:.1f}"
format_message = "{} scored {}".format(name, score)

print(fstring_message)
print(format_message)
