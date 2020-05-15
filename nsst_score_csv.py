from argparse import ArgumentParser

from nltk.translate import bleu_score
import pyter
import numpy as np
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--csv", default="output/nsst_stat_54Q0.csv")
    args = parser.parse_args()
    print(args.csv)

    refs = []
    hyps = []
    with open(args.csv, 'r') as csv:
        for line in csv:
            _, _, _, hyp, _, ref, _, _, _ = line.split(';')
            if not (len(hyp) and len(ref)):
                continue
            refs.append([[t for t in ref.split(' ') if len(t)]])
            hyps.append([t for t in hyp.split(' ') if len(t)])

    bleu = bleu_score.corpus_bleu(refs, hyps)
    print(f"Bleu: {bleu}")

    ter = np.mean([pyter.ter(hyp, ref[0]) for hyp, ref in zip(hyps, refs)])
    print(f"TER: {ter}")
