import hmmlearn
from hmmlearn import base
from HMM import MultiThreadFit
import pickle
import numpy as np
from tqdm import tqdm
import europarl_dataloader as e_dl
import tensorboardX

assert (hmmlearn.__version__ >= "0.2.3")


class TbXMonitor(base.ConvergenceMonitor):
    def __init__(self, tol, n_iter, name, model: base._BaseHMM):
        super(TbXMonitor, self).__init__(tol, n_iter, False)
        self.model = model
        self.log = tensorboardX.SummaryWriter("runs/" + name)
        self.iter = 0

    def report(self, logprob):
        """Reports to Tesorboard.

        Parameters
        ----------
        logprob : float
            The log probability of the data as computed by EM algorithm
            in the current iteration.
        """
        if self.history:
            delta = logprob - self.history[-1]
            self.log.add_scalar("delta", delta, global_step=self.iter)
        self.log.add_scalar("logprob", logprob, global_step=self.iter)
        self.history.append(logprob)

        self.log.add_image("transmat_", (model.transmat_ / model.transmat_.max())[None, :, :], global_step=self.iter)
        self.log.add_image("startprob_", (model.startprob_ / model.startprob_.max())[None, None, :],
                           global_step=self.iter)
        self.log.add_image("emissionprob_",
                           (model.emissionprob_ / model.emissionprob_.max())[None, :, :],
                           global_step=self.iter)
        self.iter += 1


num_workers = 16
dataset_path = "output/europarl-v7.de-en.de.clean"
train_step_size = 2000  # 10
threshold = 4  # 5
n_states = 100
n_iter = 101
name = f"tss{train_step_size}_th{threshold}_nSt{n_states}_nIt{n_iter}"

if __name__ == '__main__':
    # load data
    lines = e_dl.load_clean_dataset(dataset_path)

    # select data (train, test)
    testLines = lines[:4096]
    trainLines = lines[4096::train_step_size]
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
    trainAlphabet = e_dl.create_alphabet(trainWordcount, threshold=threshold)
    testAlphabet = e_dl.create_test_alphabet(trainAlphabet, testWordcount)

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
    model = MultiThreadFit(n_components=n_states, n_iter=10, num_workers=num_workers)  # save & score every 10 iters
    model.n_features = len(trainAlphabet)
    model.transmat_ = np.random.random([model.n_components, model.n_components])
    model.startprob_ = np.asarray([1 / n_states for _ in range(n_states)])
    model.emissionprob_ = np.random.random([model.n_components, model.n_features])
    model.monitor_ = TbXMonitor(model.tol, model.n_iter, name, model)
    model.monitor_.log.add_text("Info",
                                f"{sum(model._get_n_fit_scalars_per_param()[p] for p in model.params)} "
                                f"free scalar parameters")
    model.monitor_.log.add_text("Info", f"nLinesX {len(len_X)}")
    model.monitor_.log.add_text("Info", f"nX {len(X)}")
    model.monitor_.log.add_text("Info", f"nLinesY {len(len_Y)}")
    model.monitor_.log.add_text("Info", f"nY {len(Y)}")

    # train
    while model.monitor_.iter < n_iter:
        model.monitor_._reset()
        model.fit(X, len_X)

        log, model.monitor_.log = model.monitor_.log, None
        with open("output/" + name + "__" + str(model.monitor_.iter).zfill(3) + ".pkl", 'wb') as file:
            pickle.dump(model, file)
        model.monitor_.log = log

        if model.monitor_.converged:
            print("Model Converged!")
            break

        score = model.score(Y, len_Y)
        model.monitor_.log.add_scalar("score", score, global_step=model.monitor_.iter)

    # eval
    test_prob = model.score(Y, len_Y)
    print(f"TestScore:   {test_prob}")

    # save model
    model.monitor_.log.close()
    with open("output/" + name + ".pkl", 'wb') as file:
        pickle.dump(model, file)
