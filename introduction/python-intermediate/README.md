## Python Intermediate 

This folder contains exercises from the *Python Intermediate* course at [Edube](https://www.edube.org).

## 1. Cipher
This exercise is a Caesar Cipher Implementation. The user enters text and a shift value (1–25). 
Letters are shifted accordingly and digits and symbols remain unchanged.

## 2. Days of Week
Example of Object-Oriented Programming (OOP) with custom exception handling.
This program defines a custom weekday handler that can add or subtract days.
It demonstrates encapsulation, error handling with try/except, and modular methods.

## 3. Palindrome
This program checks if a given word or phrase is a palindrome.
A palindrome reads the same backward and forward (ignoring spaces and capitalization).

## 4. Read File
Parses a plain‑text file of student scores, validates each line, and prints total points per student.
Each line must contain exactly three whitespace‑separated fields.

## 5. Sudoku
This script validates a Sudoku solution by checking whether each row and each column contains all digits from 1 to 9 exactly once.

## 6. Timer
This script defines a Timer class that simulates a 24-hour digital clock.
Initialization: A Timer object starts with specified hours, minutes, and seconds (default 00:00:00).
Formatting: Time is always displayed in HH:MM:SS format. Methods:
- next_second() → moves the clock forward by one second, handling wrap-around after 23:59:59 → 00:00:00.
- prev_second() → moves the clock backward by one second, handling wrap-around before 00:00:00 → 23:59:59.

## 7. Triangle
This script implements two classes:
**Point**
Represents a point in 2D space with private coordinates x and y.
- getx() and gety() → return coordinates.
- distance_from_xy(x, y) → returns Euclidean distance from the point to given coordinates.
- distance_from_point(point) → returns distance from the point to another Point object.
**Triangle**
Represents a triangle defined by three Point objects.
perimeter() → returns the perimeter of the triangle (sum of side lengths).
