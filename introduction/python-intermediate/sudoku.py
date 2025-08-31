# Sudoku validator (rows and columns only)

list_sudoku = [1,2,3,4,5,6,7,8,9]

# Read 9 rows of Sudoku
list_numbers = []
for i in range(9):
    row = input(f"Give row {i+1} (9 digits 1-9, no spaces): ")
    if len(row) != 9 or not row.isdigit():
        raise ValueError("Each row must be 9 digits from 1 to 9")
    list_numbers.append([int(x) for x in row])

# Check rows
sudoku = True
for row in list_numbers:
    if sorted(row) != list_sudoku:
        sudoku = False
        break

# Check columns
if sudoku:
    for col in range(9):
        column = [list_numbers[row][col] for row in range(9)]
        if sorted(column) != list_sudoku:
            sudoku = False
            break

if sudoku:
    print("Sudoku rows and columns are valid.")
else:
    print("Sudoku is invalid.")



