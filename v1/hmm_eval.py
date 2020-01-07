#!/usr/bin/env python3

from hmmlearn import hmm
import pickle
import numpy as np
from tqdm import tqdm
from sys import argv

ROOT = "output/fr-en.en/"
N_STATES = 50
N_ITER = 25

NAME = "fr-en_en__alphabet000010_S100"

MODEL_PATH = "/Users/alexanderjenke/Documents/Uni/MA/VERT-2/P-PnS/nsst/fr-en_en__alphabet000001_S50.pkl"
# argv[1]

if __name__ == '__main__':
    # load output
    print("Load output...")
    with open(ROOT + "alphabet000001", 'rb') as file:
        alphabet = pickle.load(file)
    with open(ROOT + "tokenset_alphabet000001", 'rb') as file:
        tokenset = pickle.load(file)
    with open(ROOT + "wordcount", 'rb') as file:
        wordcount = pickle.load(file)

    # prepare train output
    print("Prepare output...")
    testdata = tokenset[:2000]  # first 2000 lines as testset
    Y = [[[word] for word in line] for line in tqdm(testdata) if len(line) > 0]
    Y = np.concatenate(Y)
    len_Y = [len(line) for line in testdata if len(line) > 0]

    # traindata = tokenset[2000:len(tokenset):1000]  # .1% of remaining output as trainset
    # X = [[[word] for word in line] for line in tqdm(traindata) if len(line) > 0]
    # X = np.concatenate(X)
    # len_X = [len(line) for line in traindata if len(line) > 0]

    print("Load model...")
    # load model
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)

    print("n_features", model.n_features)
    print("transmat_", model.transmat_)
    print("startprob_", model.startprob_)
    print("emissionprob_", model.emissionprob_)

    lut = {alphabet[w]: w for w in alphabet}

    score = np.ndarray(len(testdata))
    c = 0
    for line in tqdm(testdata):
        if len(line) == 0: continue
        s = model.score([[word] for word in line], [len(line)])
        if not np.isnan(s):
            score[c] = s
            c += 1

    print(min(score), max(score), score.mean(), score.var())

    test_prob = model.score(Y, len_Y)
    print("Score(testdata) =", test_prob)
