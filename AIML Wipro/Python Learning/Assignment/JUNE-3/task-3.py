# You are tasked with creating a Python program that calculates the square root of a non-negative number entered by the user. The program should handle exceptions such as ValueError and NameError appropriately. Additionally, it should include an else block to print the square root if no exception occurs and a finally block to ensure that the program execution is completed. Write the Python program to fulfill these requirements.

import math

def calculate_square_root():
    try:
        number = float(input("Enter a non-negative number: "))
        if number < 0:
            raise ValueError("Entered number must be non-negative")
        square_root = math.sqrt(number)
    except ValueError as ve:
        print("Error:", ve)
    except NameError:
        print("Invalid input. Please enter a valid number.")
    else:
        print("Square root:", square_root)
    finally:
        print("Program execution completed.")

if __name__ == '__main__':
    calculate_square_root()