from hmmlearn import hmm
import pickle
import numpy as np
from tqdm import tqdm
import europarl_dataloader as e_dl

DATASET_PATH = "output/europarl-v7.de-en.de.clean"
N_ITER = 25


def trial(wordcount_treshold, n_states, n_iter, trainLines, testLines, trainWordcount, testWordcount, name):
    # create alphabets
    trainAlphabet = e_dl.create_alphabet(trainWordcount, threshold=wordcount_treshold)
    testAlphabet = e_dl.create_test_alphabet(trainAlphabet, testWordcount)

    print(f"\nname:        {name}\n"
          f"nIter:       {n_iter}\n"
          f"nStates:     {n_states}\n"
          f"WTC:         {wordcount_treshold}\n"
          f"nTrainLines: {len(trainLines)}\n"
          f"nTokens:     {max(trainAlphabet.values())}"
          )

    # prepare tokens
    tokenLines = [[[trainAlphabet[word]] for word in line if len(word)]
                  for line in tqdm(trainLines, desc="trainTokenSet") if
                  len(line) > 1 or (len(line) == 1 and len(line[0]))]
    len_trainTokenSet = [len(line) for line in tokenLines]
    trainTokenSet = np.concatenate(tokenLines)

    del trainLines  # free space

    tokenLines = [[[testAlphabet[word]] for word in line if len(word)]
                  for line in tqdm(testLines, desc="testTokenSet") if
                  len(line) > 1 or (len(line) == 1 and len(line[0]))]
    len_testTokenSet = [len(line) for line in tokenLines]
    testTokenSet = np.concatenate(tokenLines)

    del testLines  # free space

    # setup model
    model = hmm.MultinomialHMM(n_components=n_states, n_iter=n_iter)
    model.n_features = len(trainAlphabet)
    model.transmat_ = np.random.random([model.n_components, model.n_components])
    model.startprob_ = np.asarray([1 / n_states for _ in range(n_states)])
    model.emissionprob_ = np.random.random([model.n_components, model.n_features])
    model.monitor_.verbose = True

    # train
    model.fit(trainTokenSet, len_trainTokenSet)

    # eval
    test_prob = model.score(testTokenSet, len_testTokenSet)
    print(f"TestScore:   {test_prob}")

    # save model
    with open("output/" + name + ".pkl", 'wb') as file:
        pickle.dump(model, file)


if __name__ == '__main__':
    # load data
    lines = e_dl.load_clean_dataset(DATASET_PATH)

    # select data (train, test)
    testLines = lines[:4096]
    trainLines = lines[4096::2000]
    del lines  # free space

    # create alphabet (reduce -> threshold)
    trainWordcount = e_dl.count_words(trainLines)
    testWordcount = e_dl.count_words(testLines)

    # trials
    for nStates in [200, 300, 400, 500]:  # [10, 25, 50, 100, 1000]:
        for wct in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100]:  # ,0, 200, 500, 1000]:
            name = f"{N_ITER}IT_{len(trainLines)}nTrSet_{nStates}STATES_{wct}WCT"
            trial(wct, nStates, N_ITER, trainLines, testLines, trainWordcount, testWordcount, name)
