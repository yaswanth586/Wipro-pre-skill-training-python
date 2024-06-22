# Use Python's re module to find all occurrences of the word "Python" in a given text.

import re

def find_python_occurrences(text):
    occurrences = re.findall(r'\bPython\b', text)
    return occurrences

if __name__ == '__main__':
    text = "Python is a widely used programming language. Python is loved by many developers."

    python_occurrences = find_python_occurrences(text)
    print("Occurrences of 'Python':", python_occurrences)