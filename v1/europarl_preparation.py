import re
from tqdm import tqdm
import pickle
from sys import argv, stderr

import matplotlib.pyplot as plt

TOKEN_SPLIT_CHAR = ' '


def get_clean_dataset(file_path, lower: bool = False, remove: str = None):
    print(f"loading {file_path}")
    print("    opening file.....", end="")
    with open(file_path, 'r') as file:
        # read dataset

        print("done. \n    reading file.....", end="")
        data = file.read()

        print("done. \n    cleaning output....", end="")

        if lower:
            # use all lower case
            print("making lowercase....", end="")
            data = data.lower()

        # only allow given characters, replace everything else with a space
        data = re.sub(r"[^A-Za-z0-9áàâÁÀÂéèêÉÈÊíìîÍÌÎóòôÓÒÔúùûÚÙÛäöüÄÖÜß/\\,.?!\-—\n(){}[\] ]+", " ", data)

        # Standardize hyphens
        data = re.sub(r"—", "-", data)

        # ensure punctuation marks and brackets are later read as single tokens
        data = re.sub(r",", " , ", data)
        data = re.sub(r"\.", " . ", data)
        data = re.sub(r"!", " ! ", data)
        data = re.sub(r"\?", " ? ", data)
        data = re.sub(r"\(", " ( ", data)
        data = re.sub(r"\)", " ) ", data)
        data = re.sub(r"{", " { ", data)
        data = re.sub(r"}", " } ", data)
        data = re.sub(r"\[", " [ ", data)
        data = re.sub(r"]", " ] ", data)

        if remove is not None:
            print(f"removing {remove}...", end="")
            data = re.sub(remove, " ", data)

        # reduce multi whitespaces to a single whitespace
        data = re.sub(r" +", " ", data)

        # split output into lines
        print("done. \n    splitting output...", end="")
        lines = tuple([tuple(line.strip().split(TOKEN_SPLIT_CHAR)) for line in data.split('\n')])
        print("done.")

        return lines


def create_alphabet(lines: tuple):
    alphabet = {}
    print("create alphabet")
    for line in tqdm(lines):
        for word in line:
            if word is not '' and word not in alphabet:
                alphabet[word] = len(alphabet)
    return alphabet


def get_wordcount(lines: tuple, alphabet: dict):
    print("counting words")
    count = {word: 0 for word in alphabet}
    for line in tqdm(lines):
        for word in line:
            if word is not '':
                count[word] += 1
    return count


def reduce_alphabet(alphabet: dict, wordcount: dict, threshold=1):
    print("reducing alphabet")
    return {word: alphabet[word]+1 if wordcount[word] > threshold else 0 for word in tqdm(alphabet)}


def create_tokenset(lines: tuple, alphabet: dict):
    print("create tokenset")
    return tuple([tuple([alphabet.get(word) for word in line if word is not '']) for line in tqdm(lines)])


def reduce_lines(token_pair: tuple):
    print("reducing lines")

    # add padding so no 'out of index' occurs
    lang1 = (('',),) + token_pair[0] + (('',),)
    lang2 = (('',),) + token_pair[1] + (('',),)

    assert (len(lang1) == len(lang2))
    n_lines = len(token_pair[0])

    skip = tuple(i for i in range(n_lines)
                 if len(lang1[i]) == 1 and lang1[i][0] in [".", ""]
                 and len(lang2[i]) == 1 and lang2[i][0] in [".", ""])
    red1 = tuple(i for i in range(n_lines)
                 if len(lang1[i]) == 1 and lang1[i][0] in [".", ""])
    red2 = tuple(i for i in range(n_lines)
                 if len(lang2[i]) == 1 and lang2[i][0] in [".", ""])

    """
    red_lang1 = ()
    red_lang2 = ()
    
    for i in tqdm(range(n_lines)):
        if i in skip:
            continue

        elif i in red1:
            red_lang1 += (lang1[i - 1] + lang1[i + 1],)
            red_lang2 += (lang2[i - 1] + lang2[i] + lang2[i + 1],)

        elif i in red2:
            red_lang1 += (lang1[i - 1] + lang1[i] + lang1[i + 1],)
            red_lang2 += (lang2[i - 1] + lang2[i + 1],)

        else:
            red_lang1 += (lang1[i],)
            red_lang2 += (lang2[i],)

    (tuple(lang1[i - 1] + lang1[i + 1] if i in red1 else
           lang1[i - 1] + lang1[i] + lang1[i + 1] if i in red2 else
           lang1[i]
           for i in range(n_lines) if i not in skip),
     tuple(lang2[i - 1] + lang2[i + 1] if i in red2 else
           lang2[i - 1] + lang2[i] + lang2[i + 1] if i in red1 else
           lang2[i]
           for i in range(n_lines) if i not in skip)
     )
    """

    exclude = tuple(i - 1 for i in skip + red1 + red2) + tuple(i for i in skip + red1 + red2) + tuple(
        i + 1 for i in skip + red1 + red2)

    return (tuple(line for i, line in enumerate(lang1) if i not in exclude),
            tuple(line for i, line in enumerate(lang2) if i not in exclude))


if __name__ == '__main__':
    """
    if len(argv) != 2:
        print("Argument expected! \nUsage: python3 europarl_preparation.py EUROPARL_FILE", file=stderr)
        exit(1)

    path = argv[1]
    output = get_clean_dataset(path)
    pickle.dump(output, open(path + ".clean.pkl", 'wb'))

    alphabet = create_alphabet(output)
    pickle.dump(alphabet, open(path + ".alphabet.pkl", 'wb'))

    tokenset = create_tokenset(output, alphabet)
    pickle.dump(tokenset, open(path + ".tokenset.pkl", 'wb'))
    """

    europarl_root = "/Users/alexanderjenke/Documents/Uni/MA/VERT-2/P-PnS/europarl/output/europarl-v7."
    europarl_sets = ["de-en.en",
                     "de-en.de",
                     "fr-en.en",
                     "fr-en.fr",
                     ]

    output = "output/"  # "output/lower"

    for set in europarl_sets:
        data = get_clean_dataset(europarl_root + set, lower=False)
        with open(output + set + "/lines", 'wb') as f:
            pickle.dump(data, f)

        alphabet = create_alphabet(data)
        with open(output + set + "/alphabet", 'wb') as f:
            pickle.dump(alphabet, f)

        wordcount = get_wordcount(data, alphabet)
        with open(output + set + "/wordcount", 'wb') as f:
            pickle.dump(wordcount, f)

        tokenset = create_tokenset(data, alphabet)
        with open(output + set + "/tokenset_alphabet", 'wb') as f:
            pickle.dump(tokenset, f)

        red_a = [None]
        red_tokenset = [None]
        for i in range(1, 11):
            red_a.append(reduce_alphabet(alphabet, wordcount, threshold=i))
            with open(output + set + "/alphabet"+str(i).zfill(6), 'wb') as f:
                pickle.dump(red_a[i], f)

            red_tokenset.append(create_tokenset(data, red_a[i]))
            with open(output + set + "/tokenset_alphabet"+str(i).zfill(6), 'wb') as f:
                pickle.dump(red_tokenset[i], f)



        """
        print(counts)
        graph = [len([1 for w in wordcount[set] if wordcount[set][w] <= c])/len(wordcount[set]) for c in tqdm(counts)]
        print(graph)

        plt.figure(set)
        plt.title(set)
        plt.plot(graph, counts)
        plt.show()
        

        counts = []
        for c in wordcount.values():
            if c not in counts:
                counts.append(c)

        counts = sorted(counts)

        h = [[wordcount[set][w] for w in wordcount[set] if wordcount[set][w] <= c]
                  for c in tqdm(counts)]
        num_vocab = tuple(len(c) / len(wordcount[set]) for c in tqdm(h))

        sum_words = sum(wordcount[set][w] for w in wordcount[set])
        num_words = tuple(sum(c) / sum_words for c in tqdm(h))
        print(num_vocab)
        print(num_words)

        plt.title(set)
        plt.plot(counts, num_words, label='all Words')
        plt.plot(counts, num_vocab, label='unique Words')
        plt.legend()
        plt.xlabel('Word Occurence')
        plt.ylabel('%')
        plt.show()
        """
        # data_lower[set] = get_clean_dataset(europarl_root + set, lower=True)
        # alphabet_lower[set] = create_alphabet(data_lower[set])
        # wordcount_lower[set] = get_wordcount(data_lower[set], alphabet_lower[set])
        # red_alphabet[set] = reduce_alphabet(alphabet[set], wordcount[set])
        # red_alphabet_lower[set] = reduce_alphabet(alphabet_lower[set], wordcount_lower[set])
        # tokenset[set] = create_tokenset(output[set], alphabet[set])
        # red_tokenset[set] = create_tokenset(output[set], red_alphabet[set])
