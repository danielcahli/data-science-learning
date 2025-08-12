count = 0
list_numbers = []
list_sudoku = [1,2,3,4,5,6,7,8,9]
while count < 9:
    count += 1
    list_numbers.append(input("Give numbers from 1 to 9: ") )

list_rows =[]
for i in range(0,8):
    for d in list_numbers[i]:
        list_rows.append(int(d)) 
        if sorted(list_rows) != list_sudoku:
            sudoku = False
        else: sudoku = True

list_colums = []
for i in range(0,8):
    for j in range(0,8):
    list_colums.append(list_numbers[i].index(j)) 



