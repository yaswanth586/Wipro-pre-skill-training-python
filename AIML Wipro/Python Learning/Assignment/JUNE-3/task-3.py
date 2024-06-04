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