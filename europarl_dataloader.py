import os
import re
from sys import stdout

from tqdm import tqdm

TOKEN_SPLIT_CHAR = ' '


def load_dataset(file_path, lower: bool = False, remove: str = None) -> tuple:
    """This function loads a europarl dataset and preprocesses the lines.
    Reduces sentences to contain only allowed characters (see code below).
    Separates the data into a tuple containing the sentence data.
    Separates the sentence into a tuple containing the words and punctuations.

    :param file_path: path to the europarl file
    :param lower: bool if sentences should be converted to lowercase
    :param remove: optional regex expression to be removes from sentences
    :return: preprocessed sentences (tuple of tuples of words/punctuation)
    """
    print(f"loading {file_path}")
    print("    opening file.......", end="")
    with open(file_path, 'r') as file:
        # read dataset

        print("done. \n    reading file.......", end="")
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

        # ensure punctuation marks and brackets are later read as single tokens by adding spaces around the chars.
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
            # optionally remove data described by the given regex
            print(f"removing {remove}...", end="")
            data = re.sub(remove, " ", data)

        # reduce multiple whitespaces to a single one
        data = re.sub(r" +", " ", data)

        # split output into lines (tuple of tuples of words)
        print("done. \n    splitting output...", end="")
        lines = tuple([tuple(line.strip().split(' ')) for line in data.split('\n')])
        print("done.")

        return lines


def load_clean_dataset(file_path):
    """ This function loads preprocessed europarl data.

    :param file_path: path to preprocessed file
    :return: data in form tuple of tuples of words/punctuation
    """
    with open(file_path, 'r') as file:
        return tuple([tuple(line.strip().split(TOKEN_SPLIT_CHAR)) for line in file.read().split('\n')])


def count_words(lines: tuple):
    """This function counts the occurrence of unique words.

    :param lines: data in form tuple of tuples of words/punctuation
    :return: dict containing word:count
    """
    wordcount = {}
    for line in tqdm(lines, desc="count words"):
        for word in line:
            if word not in wordcount:
                wordcount[word] = 0
            wordcount[word] += 1
    return wordcount


def create_tokenization(wordcount: dict, threshold=None):
    """This function creates an tokenization to translate words into tokens
    (numerical representation of word processable by HMM).

    :param wordcount: wordcount of the data the tokenization should be applied to given by the function 'count_words'.
    :param threshold: optional threshold defining words to be summarized into the token '0'
                      if the wordoccurence is not above the threshold
    :return: dict containing word:token
    """
    tokenization = {}

    i = 1  # next free token (0 is reserved for collection of words defined by threshold)
    for word in tqdm(wordcount, desc="create tokenization"):
        if not len(word):  # skip empty word
            continue
        if threshold is not None and wordcount[word] <= threshold:  # word not above threshold -> token: 0
            tokenization[word] = 0
        else:  # word above threshold -> token: [next free token]
            tokenization[word] = i
            i += 1

    return tokenization


def extend_tokenization(train_tokenization: dict, test_wordcount: dict):
    """ This function extends the tokenization of the train data to be applicable to the test data
    by adding all unknown words to the collective token 0.

    :param train_tokenization: tokenization of train data to be extended
    :param test_wordcount: wordcount of test data given by the function 'count_words(test data)'.
    :return: extended tokenization containing all words of the test data (dict word:token)
    """
    tokenization = train_tokenization  # copy already known words

    for word in tqdm(test_wordcount, desc="extend tokenization"):
        if not len(word):  # skip empty word
            continue
        if word not in tokenization:  # extend tokenization by adding unknown words to the collective token 0
            tokenization[word] = 0

    return tokenization


def get_wordcount(lines: tuple, tokenization: dict):
    """This function is a faster version of count_words if the tokenization is already known
    :param lines: data in form tuple of tuples of words/punctuation
    :param tokenization: tokenization of data
    :return: dict containing word:count
    """
    print("counting words")
    count = {word: 0 for word in tokenization}
    for line in tqdm(lines):
        for word in line:
            if word is not '':
                count[word] += 1
    return count


def create_tokenset(lines: tuple, tokenization: dict):
    """This function creates a token set by translating words into tokens according to the given tokenization.

    :param lines: lines to be translated (form: tuple of tuples of words/punctuation)
    :param tokenization: tokenization containing all words occurring in lines
    :return: tuple of tuples of tokens
    """
    return tuple([tuple([tokenization.get(word) for word in line if word is not ''])
                  for line in tqdm(lines, desc="create tokenset")])


if __name__ == '__main__':
    """ The direct execution of this file pre processes a given europarl file.
    If imported the file provides the functionality to work with the pre processed data.
    """
    # add arguments to be called in the console
    from argparse import ArgumentParser

    parser = ArgumentParser(prog='python3 europarl_dataloader.py')
    parser.add_argument("-i", "--input", help="europarl data set to be preprocessed", required=True)
    parser.add_argument("-o", "--output", help="directory the proccessed data shuold be stored to, default: output",
                        default="output")
    parser.add_argument("--lower", help="make data lowercase, default: False", default=False, type=bool)
    parser.add_argument("--remove", help="optional regex expression describing data to be removed from the dataset",
                        default=None)
    parser.add_argument("--split_char", default=None,
                        help="Character used to separate words in the preprocessed data, default: ' ' (Space)", )

    args = parser.parse_args()

    output_path = args.output

    if args.split_char is not None:
        TOKEN_SPLIT_CHAR = args.split_char

    # preprocess the europarl data
    data = load_dataset(args.input, lower=args.lower, remove=args.remove)

    # store the data in a file named after the input extended by the '.clean' extension in the defined output directory
    with open(os.path.join(output_path, os.path.basename(args.input) + ".clean"), 'w') as f:
        for line in tqdm(data, desc=f"writing clean lines to {f.name}", file=stdout):
            sentence = ""
            for word in line:
                sentence += word + TOKEN_SPLIT_CHAR  # separate the words/punctuation by the defined separator
            print(sentence, file=f)  # write sentence to file
