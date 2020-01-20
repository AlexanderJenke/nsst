from hmmlearn import hmm
import pickle
import numpy as np
from tqdm import tqdm
import europarl_dataloader as e_dl
import os
import re
import matplotlib.pyplot as plt
import math
import datetime

DATASET_PATH = "output/europarl-v7.de-en.de.clean"
MODEL_PATH = "output/"

def calc_perplexity(model, tokenLines):
    perplexitys = []
    for line in tqdm(tokenLines, desc="Perplexity"):
        prob = model.score(line, [len(line)])
        # TODO -inf & < -323
        perplexitys.append(math.pow(10**prob, -1/len(line)))
    return np.mean(perplexitys), perplexitys

if __name__ == '__main__':

    # load data
    lines = e_dl.load_clean_dataset(DATASET_PATH)

    # select data (train, test)
    testLines = lines[:4096]

    testWordcount = e_dl.count_words(testLines)
    trainWordcount = {"959": e_dl.count_words(lines[4096::2000]),
                      "9581": e_dl.count_words(lines[4096::200])}

    del lines  # free space

    scores = {}

    for root, _, files in os.walk(MODEL_PATH):
        for file in sorted(files):
            if not file.endswith(".pkl"):  # or (datetime.datetime.now().timestamp() - os.path.getctime(os.path.join(root, file))) > 48*60*60:
                continue

            path = os.path.join(root, file)
            print(path)
            n_iter, n_trainlines, n_states, wct = re.sub("WCT.pkl",
                                                         "",
                                                         re.sub(r"((IT_)|(nTrSet_)|(STATES_))",
                                                                ",",
                                                                file)
                                                         ).split(",")

            trainAlphabet = e_dl.create_alphabet(trainWordcount[n_trainlines], threshold=int(wct))
            testAlphabet = e_dl.create_test_alphabet(trainAlphabet, testWordcount)

            tokenLines = [[[testAlphabet[word]] for word in line if len(word)]
                          for line in tqdm(testLines, desc="testTokenSet") if
                          len(line) > 1 or (len(line) == 1 and len(line[0]))]
            len_testTokenSet = [len(line) for line in tokenLines]
            testTokenSet = np.concatenate(tokenLines)

            # setup model
            with open(path, 'rb') as f:
                model = pickle.load(file=f)

            plt.imsave(f"output/imgs/{file}_transmat.jpg", model.transmat_, cmap=plt.cm.bone)
            plt.imsave(f"output/imgs/{file}_emissionprob.jpg", model.emissionprob_, cmap=plt.cm.bone)

            test_prob = model.score(testTokenSet, len_testTokenSet)

            print(f"\nfile:        {file}\n"
                  f"nIter:       {n_iter}\n"
                  f"nStates:     {n_states}\n"
                  f"WTC:         {wct}\n"
                  f"nTokens:     {max(trainAlphabet.values())}\n"
                  f"nTrainLines: {n_trainlines}\n"
                  f"Test Score:  {test_prob}"
                  )

            if n_iter + "_" + n_trainlines not in scores:
                scores[n_iter + "_" + n_trainlines] = {}

            if wct not in scores[n_iter + "_" + n_trainlines]:
                scores[n_iter + "_" + n_trainlines][wct] = {"n_tokens": max(trainAlphabet.values()),
                                                            "states": {}}

            scores[n_iter + "_" + n_trainlines][wct]["states"][n_states] = test_prob

    print(scores)

    with open("output/scores2.pkl", 'wb') as file:
        pickle.dump(scores, file)
