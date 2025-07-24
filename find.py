first_word = input('first word: ').lower()
second_word = input('second word: ').lower()
list_first_in_second=[]
for ch in first_word:
    list_first_in_second.append(second_word.find(ch))
if -1 in  list_first_in_second:
    print('No')
else:
    print('Yes')