# Caesar Cipher Implementation
# User enters text and a shift value (1–25). 
# Letters are shifted accordingly; digits and symbols remain unchanged.

# Prompt user for input text
text = input("Enter your message: ")

# Continuously prompt for a valid shift until user enters an integer 1–25
while True:
    shift = input("Shift from 1..25: ")
    if shift.isdigit():
        shift = int(shift)
        if 1 <= shift <= 25:
            break
    print("Invalid input. Please enter a number from 1 to 25.")

# Encrypt message using Caesar cipher
cipher = ''
for char in text:
    if not char.isalpha():
        # Non-alphabetic characters (digits, punctuation, spaces) remain unchanged
        cipher += char
    elif char.isupper():
        # Shift uppercase letters within 'A'–'Z'
        code = ord(char) + shift
        if code > ord('Z'):
            code -= 26
        cipher += chr(code)
    else:
        # Shift lowercase letters within 'a'–'z'
        code = ord(char) + shift
        if code > ord('z'):
            code -= 26
        cipher += chr(code)

# Output encrypted message
print()
print(cipher)
