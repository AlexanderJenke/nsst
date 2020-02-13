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
MODEL_PATH = "output/tss20_th4_nSt100_nIt101.pkl"

if __name__ == '__main__':
    lines = e_dl.load_clean_dataset(DATASET_PATH)
    testLines = lines[:4096]
    trainLines = lines[4096::TRAIN_STEP_SIZE]
    del lines
    trainWordcount = e_dl.count_words(trainLines)
    testWordcount = e_dl.count_words(testLines)

    print("\n")
    for i in range(1, 5):
        print("\n", i, [word for word in trainWordcount if trainWordcount[word] == i and word.islower()])
        print(i, [word for word in trainWordcount if trainWordcount[word] == i and not word.islower()])

    print("\n")
    print(i, [word for word in trainWordcount if trainWordcount[word] == 5 and word.islower()])
    print(i, [word for word in trainWordcount if trainWordcount[word] == 5 and not word.islower()])
