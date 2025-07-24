word = input("Write something.")
word = word.replace(" ", "")
word = word.lower()
print(word)
count = 0
for chr in word:
    count += 1
i = 0
p = 0
for i in range(count//2):
    if word[i]==word[count-1]:
        i += 1
        count -= 1
    else:
        p += 1
if p == 0:
    print("Palindrome")
else:
    print("not palindrome")
