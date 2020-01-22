import re
from tqdm import tqdm
from sys import stdout, argv
import os

TOKEN_SPLIT_CHAR = ' '


def load_dataset(file_path, lower: bool = False, remove: str = None):
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
        lines = tuple([tuple(line.strip().split(' ')) for line in data.split('\n')])
        print("done.")

        return lines


def load_clean_dataset(file_path):
    with open(file_path, 'r') as file:
        return tuple([tuple(line.strip().split(TOKEN_SPLIT_CHAR)) for line in file.read().split('\n')])


def count_words(lines: tuple):
    wordcount = {}
    for line in tqdm(lines, desc="count words"):
        for word in line:
            if word not in wordcount:
                wordcount[word] = 0
            wordcount[word] += 1
    return wordcount


def create_alphabet(wordcount: dict, threshold=None):
    alphabet = {}

    i = 1  # next free token (0 is reserved for collection of words by threshold)
    for word in tqdm(wordcount, desc="create alphabet"):
        if threshold is not None and wordcount[word] <= threshold:
            alphabet[word] = 0
        else:
            alphabet[word] = i
            i += 1

    return alphabet


def create_test_alphabet(train_alphabet: dict, test_wordcount: dict):
    alphabet = train_alphabet

    for word in tqdm(test_wordcount, desc="create test alphabet"):
        if word not in alphabet:
            alphabet[word] = 0

    return alphabet


def get_wordcount(lines: tuple, alphabet: dict):
    print("counting words")
    count = {word: 0 for word in alphabet}
    for line in tqdm(lines):
        for word in line:
            if word is not '':
                count[word] += 1
    return count


def create_tokenset(lines: tuple, alphabet: dict):
    print("create tokenset")
    return tuple([tuple([alphabet.get(word) for word in line if word is not '']) for line in tqdm(lines)])


if __name__ == '__main__':
    input_file = argv[1]
    output_path = "output/"
    set_name = os.path.basename(input_file)

    data = load_dataset(input_file)
    with open(output_path + set_name + ".clean", 'w') as f:
        for line in tqdm(data, desc=f"writing clean lines to {f.name}", file=stdout):
            sentence = ""
            for word in line:
                sentence += word + " "
            print(sentence, file=f)  # write sentence to file
