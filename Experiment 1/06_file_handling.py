import os

folder_name = "demo_files"
file_name = "notes.txt"

os.makedirs(folder_name, exist_ok=True)
text_path = folder_name + "/" + file_name

write_mode = "w"
read_mode = "r"
append_mode = "a"

line_one = "Line 1\n"
line_two = "Line 2\n"
line_three = "Line 3\n"
line_four = "Line 4\n"

with open(text_path, write_mode) as f:
    f.write(line_one)
    f.write(line_two)
    f.write(line_three)

with open(text_path, read_mode) as f:
    for line in f:
        clean_line = line.strip()
        print(clean_line)

with open(text_path, append_mode) as f:
    f.write(line_four)

with open(text_path, read_mode) as f:
    file_content = f.read()

print(file_content)
