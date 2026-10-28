def linear_search(items, target):
    found_index = -1
    for index, value in enumerate(items):
        if value == target:
            found_index = index
            break
    return found_index

def bubble_sort(items):
    sorted_items = items.copy()
    length = len(sorted_items)
    for i in range(length):
        for j in range(length - i - 1):
            current_value = sorted_items[j]
            next_value = sorted_items[j + 1]
            if current_value > next_value:
                sorted_items[j] = next_value
                sorted_items[j + 1] = current_value
    return sorted_items

def is_prime(n):
    prime = True
    if n < 2:
        prime = False
    else:
        upper_bound = int(n ** 0.5) + 1
        for i in range(2, upper_bound):
            if n % i == 0:
                prime = False
                break
    return prime

numbers = [5, 2, 9, 1, 5, 6]
search_target = 9

prime_start = 2
prime_stop = 30

search_result = linear_search(numbers, search_target)
sorted_numbers = bubble_sort(numbers)
primes = [n for n in range(prime_start, prime_stop) if is_prime(n)]

print(search_result)
print(sorted_numbers)
print(primes)
