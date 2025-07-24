word_1 = input('Write the first text: ').replace(" ", "").lower()
word_2 = input('Write the second text: ').replace(" ", "").lower()

if sorted(word_1) == sorted(word_2):
    print("Anagrams")
else:
    print("Not anagrams")    

         