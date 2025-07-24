num = input('whats your birthday (MMDDAAAA)?')
while int(num)>9:
    total = 0
    for ch in num:
        total += int(ch)
    num = str(total)

print(total)