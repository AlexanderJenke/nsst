from argparse import ArgumentParser

from tqdm import tqdm

import NSST
from nsst_translate import best_transition_sequence

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--nsst_file", default="output/nsst_tss20_th4_nSt54_Q0.pkl", help="nsst file")
    parser.add_argument("--src_lang", default="output/europarl-v7.de-en.de.clean")
    parser.add_argument("--tgt_lang", default="output/europarl-v7.de-en.en.clean")
    parser.add_argument("--output", default=f"output/nsst_stat54Q0.csv")
    args = parser.parse_args()
    args.enforce_n_reg = False
    args.enforce_n_final_reg = False

    # load NSST
    nsst = NSST.NSST()
    nsst.load(args.nsst_file, doCheckRules=False)
    args.nsst = nsst

    # open files
    src_file = open(args.src_lang, 'r')
    tgt_file = open(args.tgt_lang, 'r')
    output_file = open(args.output, 'w')

    # iterate over sentences, first 4096 -> test sentences
    for src, tgt, _ in tqdm(list(zip(src_file, tgt_file, range(4096))), desc="Processing sentences"):
        # remove line breaks
        src = src[:-1]
        tgt = tgt[:-1]

        # try to translate
        try:
            # prepare tokenisations
            token_src = [nsst.tokenization_src[word] if word in nsst.tokenization_src else 0
                         for word in src.split(" ") if len(word)]

            token_tgt = [nsst.tokenization_tgt[word] if word in nsst.tokenization_tgt else 0
                         for word in tgt.split(" ") if len(word)]

            # run nsst
            args.input = src
            args.token_src = token_src
            result = best_transition_sequence(args)

            # get best result
            pred = sorted((k for k in result
                           if ('Qf' in args.nsst_file or not args.enforce_n_final_reg or len(k[1]) == 1)
                           and ('Q0' in args.nsst_file or k[0] == -1)
                           ),
                          key=lambda x: x[2],
                          reverse=True)[0]

            n_res = len(result)
            q, reg, prob = pred

            # write to csv
            if not len(reg):  # catch empty registers
                continue

            token_pred = [w for w in reg[0].split(' ') if len(w)]

            pred_str = ""
            for t in token_pred:
                pred_str += f"{nsst.tokenization_tgt_lut[int(t)]} "

            token_src_str = ""
            for t in token_src:
                token_src_str += f"{t} "

            token_tgt_str = ""
            for t in token_tgt:
                token_tgt_str += f"{t} "

            token_pred_str = ""
            for t in token_pred:
                token_pred_str += f"{t} "

            print(f"{src};{token_src_str[:-1]};"
                  f"{tgt};{token_tgt_str[:-1]};"
                  f"{pred_str};{token_pred_str[:-1]};"
                  f"{prob};{len(reg)};{n_res}",
                  file=output_file)

            output_file.flush()
        except RuntimeError:
            pass

    # close files
    src_file.close()
    tgt_file.close()
    output_file.close()
