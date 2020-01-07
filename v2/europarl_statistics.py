import v2.europarl_dataloader as e_dl
from tqdm import tqdm
import tensorboardX
import os

if __name__ == '__main__':
    datapath = "output/europarl-v7.de-en.de.clean"

    lines = e_dl.load_clean_dataset(datapath)
    log = tensorboardX.SummaryWriter("output/tbX/" + os.path.basename(datapath), max_queue=999, flush_secs=2)

    wordcount = e_dl.count_words(lines)

    numWuedWords = sum(wordcount.values())
    numUniqueWords = len(wordcount)
    maxWordcount = max(wordcount.values())
    minWordcount = min(wordcount.values())

    valueset = sorted(list(set(wordcount.values())))
    for i in tqdm(valueset, desc="creating statistics"):
        uniqueWordsBelowThreshold_value = sum(1 if wordcount[word] <= i else 0 for word in wordcount)
        usedWordsBelowThreshold_value = sum(wordcount[word] for word in wordcount if wordcount[word] <= i)

        log.add_scalar("uniqueWordsBelowThreshold",
                       uniqueWordsBelowThreshold_value,
                       global_step=i)

        log.add_scalar("usedWordsBelowThreshold",
                       usedWordsBelowThreshold_value,
                       global_step=i)

        log.add_scalar("usedWordsPerUniqueWordBelowThreshold",
                       usedWordsBelowThreshold_value / uniqueWordsBelowThreshold_value,
                       global_step=i)
