def sum_of_odd_digits(number):
    odd_sum = 0
    while number > 0:
        digit = number % 10
        if digit % 2 != 0:
            odd_sum += digit
        number //= 10
    return odd_sum

def sum_of_even_digits(number):
    even_sum = 0
    while number > 0:
        digit = number % 10
        if digit % 2 == 0:
            even_sum += digit
        number //= 10
    return even_sum

def difference_of_sums_of_odd_and_even_digits(number):
    odd_sum = sum_of_odd_digits(number)
    even_sum = sum_of_even_digits(number)
    return odd_sum - even_sum

if __name__ == '__main__':
    number = 123456
    result = difference_of_sums_of_odd_and_even_digits(number)
    print("Difference between sums of odd and even digits:", result)