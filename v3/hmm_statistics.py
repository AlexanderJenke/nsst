import hmmlearn
from hmmlearn import base
from HMM import MultiThreadFit
import pickle
import numpy as np
from tqdm import tqdm
import europarl_dataloader as e_dl
from hmm_training import TbXMonitor

DATASET_PATH = "output/europarl-v7.de-en.de.clean"
TRAIN_STEP_SIZE = 20
THRESHOLD = 4
MODEL_PATH = "output/tss20_th4_nSt200_nIt101.pkl"

if __name__ == '__main__':
    print(MODEL_PATH)
    lines = e_dl.load_clean_dataset(DATASET_PATH)
    testLines = lines[:4096]
    trainLines = lines[4096::TRAIN_STEP_SIZE]
    del lines
    trainWordcount = e_dl.count_words(trainLines)
    testWordcount = e_dl.count_words(testLines)
    trainAlphabet = e_dl.create_alphabet(trainWordcount, threshold=THRESHOLD)
    testAlphabet = e_dl.create_test_alphabet(trainAlphabet, testWordcount)

    '''
    # prepare tokens
    lines_X = [[[trainAlphabet[word]] for word in line if len(word)]
               for line in tqdm(trainLines, desc="trainTokenSet") if
               len(line) > 1 or (len(line) == 1 and len(line[0]))]
    len_X = [len(line) for line in lines_X]
    X = np.concatenate(lines_X)

    del trainLines  # free space

    lines_Y = [[[testAlphabet[word]] for word in line if len(word)]
               for line in tqdm(testLines, desc="testTokenSet") if
               len(line) > 1 or (len(line) == 1 and len(line[0]))]
    len_Y = [len(line) for line in lines_Y]
    Y = np.concatenate(lines_Y)

    del testLines  # free space
    # '''

    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
    """ :type model: MultiThreadFit"""

    alphabet_lut = {testAlphabet[i]: i for i in testAlphabet}
    alphabet_lut[0] = "PLACEHOLDER"

    states = {}

    # words per state unique words
    for i in range(model.emissionprob_.shape[1]):
        s = model.emissionprob_[:, i].argmax()
        if s not in states:
            states[s] = []
        states[s].append(alphabet_lut[i])

    for i in sorted(states):
        print(i, states[i])

    input("WAITING...\n\n")

    # words per state
    for i, prob in enumerate(model.emissionprob_[:]):
        print(f"{i}: ", end="")
        breaker = False
        for j, s in sorted(zip(prob, range(len(prob))), key=lambda x: x[0], reverse=True)[:20]:
            if j < prob.max() * 0.001 and not breaker:
                breaker = True
                print(" |  ", end="")
            print(f"{alphabet_lut[s]}, ", end="")

        print("")

    input("WAITING...\n\n")
    sp = {k: model.startprob_[k] for k in range(len(model.startprob_))}
    print({k: sp[k] for k in sorted(sp, key=lambda x: sp[x], reverse=True)})

    sentences = {}
    likelyhoods = []

    for sentence in testLines:
        tokens = [[testAlphabet[word]] for word in sentence if len(word)]
        if not len(tokens):
            continue
        prob, stateprob = model.score_samples(tokens, [len(tokens)])
        likelyhoods.append(10 ** (prob * (1.0 / len(tokens))))
        sstate = stateprob[0].argmax()
        if sstate not in sentences:
            sentences[sstate] = []
        sentences[sstate].append(sentence)

    print("Likelyhood", np.sum(likelyhoods))

    # for k in sorted(sentences.keys()):
    for k in sorted(sp, key=lambda x: sp[x], reverse=True):
        print(k)
        if k in sentences:
            for l in sentences[k][:10]:
                for w in l:
                    print(f"{w} ", end="")
                print("")
        else:
            print(f"{k} not in sentences!")

        print("\n\n")
