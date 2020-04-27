import os
import pickle
from argparse import ArgumentParser

import hmmlearn
import numpy as np
from tqdm import tqdm

import europarl_dataloader as e_dl
from HMM import MultiThreadFit, TbXMonitor

if hmmlearn.__version__ < "0.2.3":  # check for hmmlerarn version
    raise ImportError(f"The minimum required version of 'hmmlearn' is 0.2.3 but found {hmmlearn.__version__}!")

# add arguments to be called in the console
parser = ArgumentParser(prog='python3 hmm_training.py')
parser.add_argument("-i", "--input", help="preprocessed europarl dataset of source language", required=True)
parser.add_argument("-o", "--output", default="output",
                    help="output directory, default: output")
parser.add_argument("-tss", "--train_step_size", default=20, type=int,
                    help="Step size over train sentences, for n every n-th sentence is used. Default: 20")
parser.add_argument("-th", "--threshold", default=4, type=int,
                    help="Threshold defining maximum word count of words to be summarized in collective in token 0,"
                         " default: 4")
parser.add_argument("-s", "--states", default=128, type=int, help="Number of states in the HMM, default: 128")
parser.add_argument("-it", "--iterations", default=100, type=int,
                    help="Maximum number of iterations for training, default: 100 ")
parser.add_argument("-ips", "--iterations_per_score", default=3, type=int,
                    help="Numper of iterations between scoring & saving the current model, default: 3")
parser.add_argument("-m", "--model", default=None, help="Optional path to model file to be loaded")
parser.add_argument("--num_workers", default=16, type=int,
                    help="Maximal number of threads running in parallel while training, default: 16")
args = parser.parse_args()

name = f"hmm_{os.path.basename(args.input)}_tss{args.train_step_size}_th{args.threshold}_nSt{args.states}_nIt{args.iterations}"

# load data
lines = e_dl.load_clean_dataset(args.input)

# select data (train, test)
testLines = lines[:4096]  # first 4096 lines are always test data
trainLines = lines[4096::args.train_step_size]  # the rest is used as train data according to tss
del lines  # free space

# count word occurrences
trainWordcount = e_dl.count_words(trainLines)
testWordcount = e_dl.count_words(testLines)

# create tokenizations & save them to the output directory
trainTokenization = e_dl.create_tokenization(trainWordcount, threshold=args.threshold)
testTokenization = e_dl.extend_tokenization(trainTokenization, testWordcount)
with open(os.path.join(args.output, f"tokenization_tss{args.train_step_size}_th{args.threshold}.pkl"), 'wb') as file:
    pickle.dump(testTokenization, file)

# prepare tokens to be passed to the HMM
lines_X = [[[trainTokenization[word]] for word in line if len(word)]
           for line in tqdm(trainLines, desc="trainTokenSet") if
           len(line) > 1 or (len(line) == 1 and len(line[0]))]
len_X = [len(line) for line in lines_X]
X = np.concatenate(lines_X)

del trainLines  # free space

lines_Y = [[[testTokenization[word]] for word in line if len(word)]
           for line in tqdm(testLines, desc="testTokenSet") if
           len(line) > 1 or (len(line) == 1 and len(line[0]))]
len_Y = [len(line) for line in lines_Y]
Y = np.concatenate(lines_Y)

del testLines  # free space

# setup the hidden markov model
model = MultiThreadFit(n_components=args.states, n_iter=args.iterations_per_score,
                       num_workers=args.num_workers)  # save & score every ips iterations
model.n_features = len(trainTokenization)  # number of features = number of tokens
model.transmat_ = np.random.random([model.n_components, model.n_components])  # init transition probability matrix
model.startprob_ = np.asarray([1 / args.states for _ in range(args.states)])  # init start probability matrix
model.emissionprob_ = np.random.random([model.n_components, model.n_features])  # init emission probability matrix
model.monitor_ = TbXMonitor(model.tol, args.iterations, name, model)  # init model monitor writing logs to tensorboard

# load pre trained model if given in arguments
if args.model is not None:
    log = model.monitor_.log
    with open(args.model, 'rb') as file:
        model = pickle.load(file)
    model.monitor_.log = log
else:
    model.monitor_._reset()

# add additional information to tensorboard log
model.monitor_.log.add_text("Info",
                            f"{sum(model._get_n_fit_scalars_per_param()[p] for p in model.params)} "
                            f"free scalar parameters")
model.monitor_.log.add_text("Info", f"nLinesX {len(len_X)}")
model.monitor_.log.add_text("Info", f"nX {len(X)}")
model.monitor_.log.add_text("Info", f"nLinesY {len(len_Y)}")
model.monitor_.log.add_text("Info", f"nY {len(Y)}")

# train the model
while model.monitor_.iter <= args.iterations:
    model.fit(X, len_X)  # train the model for ips iterations on train data

    # save the current model
    log, model.monitor_.log = model.monitor_.log, None  # dont save the tensorboard log (not savable by pickle)
    with open(os.path.join(args.output, name + "__" + str(model.monitor_.iter).zfill(3) + ".pkl"), 'wb') as file:
        pickle.dump(model, file)
    model.monitor_.log = log

    score = model.score(Y, len_Y)  # score the current model on test data
    model.monitor_.log.add_scalar("score", score, global_step=model.monitor_.iter)  # add score to the log

    if model.monitor_.converged:  # check if the termination criterion is fulfilled
        print("Model Converged!")
        break

# save final model
model.monitor_.log.close()
with open(os.path.join(args.output, name + ".pkl"), 'wb') as file:
    pickle.dump(model, file)
