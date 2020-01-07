from hmmlearn import hmm
import pickle
import numpy as np
from tqdm import tqdm

ROOT = "output/fr-en.en/"
N_STATES = 50
N_ITER = 25

NAME = "fr-en_en__alphabet000001_S50"

if __name__ == '__main__':
    # load output
    print("Load output...")
    with open(ROOT + "alphabet000001", 'rb') as file:
        alphabet = pickle.load(file)
    with open(ROOT + "tokenset_alphabet000001", 'rb') as file:
        tokenset = pickle.load(file)

    # prepare train output
    print("Prepare output...")
    testdata = tokenset[:2000]  # first 2000 lines as testset
    Y = [[[word] for word in line] for line in tqdm(testdata) if len(line) > 0]
    Y = np.concatenate(Y)
    len_Y = [len(line) for line in testdata if len(line) > 0]

    traindata = tokenset[2000:len(tokenset):1000]  # .1% of remaining output as trainset
    X = [[[word] for word in line] for line in tqdm(traindata) if len(line) > 0]
    X = np.concatenate(X)
    len_X = [len(line) for line in traindata if len(line) > 0]

    print("Setup model...")
    # setup model
    model = hmm.MultinomialHMM(n_components=N_STATES, n_iter=N_ITER)
    model.n_features = len(alphabet)
    model.transmat_ = np.random.random([model.n_components, model.n_components])
    model.startprob_ = np.asarray([1 / N_STATES for _ in range(N_STATES)])
    model.emissionprob_ = np.random.random([model.n_components, model.n_features])
    model.monitor_.verbose = True

    # fit model
    print("Fitting...")
    model.fit(X, len_X)

    test_prob = model.score(Y, len_Y)
    print("Score(testdata) =", test_prob)

    # save model
    with open(NAME + ".pkl", 'wb') as file:
        pickle.dump(model, file)

    with open(NAME + ".log", 'w') as log:
        print("ROOT", ROOT, file=log)
        print("N_STATES", N_STATES, file=log)
        print("N_ITER", N_ITER, file=log)
        print("len(traindata)", len(len_X), file=log)
        print("Score(testdata) =", test_prob, file=log)
