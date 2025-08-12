# Caesar cipher.
text = input("Enter your message: ")

while True:
    shift = input("Shift from 1..25: ")
    if shift.isdigit():
        shift = int(shift)
        if 1 <= shift <= 25:
            break
    print("Invalid input. Please enter a number from 1 to 25.")
   
cipher = ''
for char in text:
    if not char.isalpha():
        cipher += char
    elif char.isupper():
        code = ord(char) + int(shift)
        if code > ord('Z'):
            code = code-26
        cipher += chr(code)
    else:
        code = ord(char) + int(shift)
        if code > ord('z'):
            code = code-26
        cipher += chr(code)
print()
print(cipher)
