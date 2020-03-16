import pickle

import hmmlearn
import numpy as np
from tqdm import tqdm

import europarl_dataloader as e_dl
from HMM import MultiThreadFit, TbXMonitor

assert (hmmlearn.__version__ >= "0.2.3")

NUM_WORKERS = 16
DATASET_PATH = "output/europarl-v7.de-en.de.clean"
TRAIN_STEP_SIZE = 20  # 10
THRESHOLD = 4  # 5
N_STATES = 128  # 54
N_ITER = 101
N_ITER_PER_SCORE = 3
name = f"tss{TRAIN_STEP_SIZE}_th{THRESHOLD}_nSt{N_STATES}_nIt{N_ITER}"

MODEL_PATH = None
# MODEL_PATH = "/Users/alexanderjenke/Documents/Uni/MA/VERT-2/P-PnS/nsst/v3/output/tss200_th4_nSt10_nIt101.pkl"

if __name__ == '__main__':
    # load data
    lines = e_dl.load_clean_dataset(DATASET_PATH)

    # select data (train, test)
    testLines = lines[:4096]
    trainLines = lines[4096::TRAIN_STEP_SIZE]
    del lines  # free space

    # create alphabet (reduce -> threshold)
    trainWordcount = e_dl.count_words(trainLines)
    testWordcount = e_dl.count_words(testLines)

    '''
    for i in range(1, 21):
        words = [w for w in trainWordcount if trainWordcount[w] == i]
        kap = [w for w in trainWordcount if w[0].isupper()]
        print(i, len(words), len(kap), f"{len(kap) / len(words)*100:.2f}%", words[:100], "\n")
    exit()
    # '''

    # create alphabets
    trainAlphabet = e_dl.create_alphabet(trainWordcount, threshold=THRESHOLD)
    testAlphabet = e_dl.create_test_alphabet(trainAlphabet, testWordcount)
    with open(f"output/alphabet_tss{TRAIN_STEP_SIZE}_th{THRESHOLD}.pkl", 'wb') as file:
        pickle.dump(testAlphabet, file)

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

    # setup model
    model = MultiThreadFit(n_components=N_STATES, n_iter=N_ITER_PER_SCORE,
                           num_workers=NUM_WORKERS)  # save & score every N_IER_PER_SCORE iters
    model.n_features = len(trainAlphabet)
    model.transmat_ = np.random.random([model.n_components, model.n_components])
    model.startprob_ = np.asarray([1 / N_STATES for _ in range(N_STATES)])
    model.emissionprob_ = np.random.random([model.n_components, model.n_features])
    model.monitor_ = TbXMonitor(model.tol, N_ITER, name, model)

    if MODEL_PATH is not None:
        log = model.monitor_.log
        with open(MODEL_PATH, 'rb') as file:
            model = pickle.load(file)
        model.monitor_.log = log
    else:
        model.monitor_._reset()

    model.monitor_.log.add_text("Info",
                                f"{sum(model._get_n_fit_scalars_per_param()[p] for p in model.params)} "
                                f"free scalar parameters")
    model.monitor_.log.add_text("Info", f"nLinesX {len(len_X)}")
    model.monitor_.log.add_text("Info", f"nX {len(X)}")
    model.monitor_.log.add_text("Info", f"nLinesY {len(len_Y)}")
    model.monitor_.log.add_text("Info", f"nY {len(Y)}")

    # train
    while model.monitor_.iter < N_ITER:
        model.fit(X, len_X)

        log, model.monitor_.log = model.monitor_.log, None
        with open("output/" + name + "__" + str(model.monitor_.iter).zfill(3) + ".pkl", 'wb') as file:
            pickle.dump(model, file)
        model.monitor_.log = log

        score = model.score(Y, len_Y)
        model.monitor_.log.add_scalar("score", score, global_step=model.monitor_.iter)

        if model.monitor_.converged:
            print("Model Converged!")
            break

    # save model
    model.monitor_.log.close()
    with open("output/" + name + ".pkl", 'wb') as file:
        pickle.dump(model, file)
