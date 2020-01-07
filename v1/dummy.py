import pickle
from tqdm import tqdm

wen = pickle.load(open("output/fr-en.en/wordcount", 'rb'))

dfr = pickle.load(open("output/fr-en.fr/tokenset_longAlphabet", 'rb'))
den = pickle.load(open("output/fr-en.en/tokenset_longAlphabet", 'rb'))

afr = pickle.load(open("output/fr-en.fr/alphabet", 'rb'))
aen = pickle.load(open("output/fr-en.en/alphabet", 'rb'))

from v1 import europarl_preparation as ep

red_alphabet = ep.reduce_alphabet(aen, wen, threshold=1)


lut = {aen[k]: k for k in aen}

c = 0
for line in den[:2000]:
    for token in line:
        word = lut[token]
        if wen[word] <= 1:
            c += 1
            print(word, wen[word])
print(c)

exit()

l_fr = [i for i, row in enumerate(tqdm(dfr)) if (len(row) == 1 and row[0] in [afr['.'], afr[',']]) or len(row) == 0]
l_en = [i for i, row in enumerate(tqdm(den)) if (len(row) == 1 and row[0] in [aen['.'], aen[',']]) or len(row) == 0]

a = set(l_fr)
b = set(l_en)

print(len(a | b))
print(len(a & b))

print(l_fr)
print(l_en)

fr = pickle.load(open("output/fr-en.fr/lines", 'rb'))
en = pickle.load(open("output/fr-en.en/lines", 'rb'))

for i in range(1000):
    print(i)
    print(fr[i])
    print(en[i])
    print()

if __name__ == '__main__':
    pass
