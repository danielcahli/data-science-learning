
src = open(r"C:\Users\danie\py\progs\newtext.txt", encoding='utf-8')

dct = {}
for line in src:
    for ch in line.lower():
        if ch.isalpha():  # só letras de a–z
            if ch in dct:
                dct[ch] += 1
            else:
                dct[ch] = 1
nsrc = open(src.name + 'hist', 'wt')
src.close()

ordenado = dict(sorted(dct.items(), key=lambda item: item[1], reverse=True))

for chave, valor  in ordenado.items():
    nsrc.write(str(chave) + " -> " + str(valor) + '\n')
nsrc.close()

