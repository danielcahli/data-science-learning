# Palindrome Checker
# This program checks if a given word or phrase is a palindrome.
# A palindrome reads the same backward and forward (ignoring spaces and capitalization).

# Get user input, remove spaces, and convert to lowercase
word = input("Write something.")
word = word.replace(" ", "")
word = word.lower()
print(word)

# Itarate for the word and check if the first word is equal the last one,
# and if the second word is equal the last - 1 and so on. If all the checks
# are true will print palindrome, else not palindrome

# Count total characters in the processed word
count = 0
for chr in word:
    count += 1

# Compare characters from the beginning and the end moving inward
# If all pairs match, the word is a palindrome
i = 0
p = 0  # mismatch counter
for i in range(count//2):  # only need to check halfway
    if word[i]==word[count-1]:
        i += 1
        count -= 1
    else:
        p += 1 # mismatch found

# Output result based on mismatch counter
if p == 0:
    print("Palindrome")
else:
    print("not palindrome")
