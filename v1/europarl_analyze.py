from hmmlearn import hmm
import pickle
import numpy as np
import os
import tensorboardX
from tqdm import tqdm

if __name__ == '__main__':
    europarl_sets = ["de-en.en",
                     "de-en.de",
                     "fr-en.en",
                     "fr-en.fr",
                     ]

    for dataset in europarl_sets:
        print(dataset)
        log = tensorboardX.SummaryWriter("output/tensorboardX/" + dataset, max_queue=999, flush_secs=2)

        with open("output/" + dataset + "/wordcount", 'rb') as f:
            wordcount = pickle.load(f)

        counts = []
        for c in wordcount.values():
            if c not in counts:
                counts.append(c)

        counts = sorted(counts)

        sum_wordcount = sum(wordcount[w] for w in wordcount)
        len_wordcount = len(wordcount)

        log.add_histogram("WordCounts", [wordcount[w] for w in wordcount])

        for i in tqdm(counts):
            h = [wordcount[w] for w in wordcount if wordcount[w] <= i]
            log.add_scalar("coveredByTheresholdOverUniqueWords", len(h) / len_wordcount, global_step=i)
            log.add_scalar("coveredByTheresholdOverAllWords", sum(h) / sum_wordcount, global_step=i)
            log.add_scalar("coveredWordsPerCoveredUniqueWords",
                           len(h) / len_wordcount,
                           global_step=(sum(h) * 10000)//sum_wordcount)

            log.add_scalar("coveredUniqueWordsPerCoveredWords",
                           sum(h) / sum_wordcount,
                           global_step=(len(h) * 10000) // len_wordcount)

        log.close()

    for dataset in europarl_sets:
        print("lower_"+dataset)
        log = tensorboardX.SummaryWriter("output/tensorboardX/lower_" + dataset, max_queue=999, flush_secs=2)

        with open("output/lower/" + dataset + "/wordcount", 'rb') as f:
            wordcount = pickle.load(f)

        counts = []
        for c in wordcount.values():
            if c not in counts:
                counts.append(c)

        counts = sorted(counts)

        sum_wordcount = sum(wordcount[w] for w in wordcount)
        len_wordcount = len(wordcount)

        for i in tqdm(counts):
            h = [wordcount[w] for w in wordcount if wordcount[w] <= i]
            log.add_scalar("coveredByTheresholdOverUniqueWords", len(h) / len_wordcount, global_step=i)
            log.add_scalar("coveredByTheresholdOverAllWords", sum(h) / sum_wordcount, global_step=i)
            log.add_scalar("coveredWordsPerCoveredUniqueWords",
                           len(h) / len_wordcount,
                           global_step=(sum(h) * 10000)//sum_wordcount)

            log.add_scalar("coveredUniqueWordsPerCoveredWords",
                           sum(h) / sum_wordcount,
                           global_step=(len(h) * 10000) // len_wordcount)

        log.close()

'''
        with open("output/" + dataset + "/lines", 'wb') as f:
            output = pickle.load(f)

        with open("output/" + dataset + "/alphabet", 'wb') as f:
            alphabet = pickle.load(f)

        with open("output/" + dataset + "/tokenset_longAlphabet", 'wb') as f:
            tokenset = pickle.load(f)
'''
'''
    # load output

    with open(FILE1 + ".alphabet.pkl", 'rb') as file:
        alphabet1 = pickle.load(file)
    with open(FILE1 + ".tokenset.pkl", 'rb') as file:
        tokenset1 = pickle.load(file)
    with open(FILE2 + ".alphabet.pkl", 'rb') as file:
        alphabet2 = pickle.load(file)
    with open(FILE2 + ".tokenset.pkl", 'rb') as file:
        tokenset2 = pickle.load(file)

    lut1 = sorted(alphabet1, key=lambda x: alphabet1[x])
    lut2 = sorted(alphabet2, key=lambda x: alphabet2[x])

    single = [lut1[i[0]] for i in tokenset1 if len(i) == 1]
    single_d = {key: single.count(key) for key in set(single)}
    for key in sorted(single_d, key=lambda x: single_d[x], reverse=True):
        print(key, single_d[key])
    exit(0)

    diff = [abs(len(tokenset1[i]) - len(tokenset2[i])) for i in range(min(len(tokenset1), len(tokenset2)))]
    print(len(diff), np.min(diff), np.max(diff), np.mean(diff), np.var(diff), np.std(diff))

    diff = [abs(len(tokenset1[i]) - len(tokenset2[i]))
            for i in range(min(len(tokenset1), len(tokenset2)))
            if len(tokenset1[i]) == 0
            or (len(tokenset1[i]) == 1
                and tokenset1[i][0] in [alphabet1['.'], alphabet1['!'], alphabet1['?']])
            or len(tokenset1[i - 1]) == 0
            or (len(tokenset1[i - 1]) == 1
                and tokenset1[i - 1][0] in [alphabet1['.'], alphabet1['!'], alphabet1['?']])
            or len(tokenset1[i + 1]) == 0
            or (len(tokenset1[i + 1]) == 1
                and tokenset1[i + 1][0] in [alphabet1['.'], alphabet1['!'], alphabet1['?']])
            ]
    print(len(diff), np.min(diff), np.max(diff), np.mean(diff), np.var(diff), np.std(diff))

    empty1 = [(i, tokenset1[i]) for i in range(len(tokenset1))
              if len(tokenset1[i]) == 0
              or (len(tokenset1[i]) == 1
                  and tokenset1[i][0] in [alphabet1['.'], alphabet1['!'], alphabet1['?']])]

    empty2 = [(i, tokenset2[i]) for i in range(len(tokenset2))
              if len(tokenset2[i]) == 0
              or (len(tokenset2[i]) == 1
                  and tokenset2[i][0] in [alphabet2['.'], alphabet2['!'], alphabet2['?']])]

    print(len(empty2)*3)
    '''
