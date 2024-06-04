choice = float(input("Enter your choice: "))
print(choice)
match choice:
    case 1:
        print("ONE")
    case 2:
        print("TWO")
    case 3:
        print("THREE")
    case _:
        print("NOT A VALID CHOICE")
