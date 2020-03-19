import pickle

import numpy as np
from tqdm import tqdm

import NSST
import europarl_dataloader as e_dl
from HMM import TbXMonitor

assert (TbXMonitor is not None)  # to keep import

alignment_file = "output/forward.N.tss20.align"
paired_language_file = "output/europarl-v7.de-en.tss20.paired"
model_file = "output/tss20_th4_nSt200_nIt101.pkl"
alphabet_file = "output/alphabet_tss20_th4.pkl"

# load hmm
with open(model_file, 'rb') as file:
    model = pickle.load(file)
""" :type model: MultiThreadFit"""

# load alphabet used with hmm
with open(alphabet_file, 'rb') as file:
    alphabet_src = pickle.load(file)


# create alphabet for target language & extend source alphabet if needed
with open(paired_language_file, 'r') as sentence_pairs:
    split_sentence_pairs = list(s[1:-2].split(" ||| ") for s in sentence_pairs)
    wordcount_src = e_dl.count_words((w for w in s[0].split(" ")) for s in split_sentence_pairs)
    wordcount_tgt = e_dl.count_words((w for w in s[1].split(" ")) for s in split_sentence_pairs)
    alphabet_tgt = e_dl.create_alphabet(wordcount_tgt)
    alphabet_extended_src = e_dl.create_test_alphabet(alphabet_src, wordcount_src)

# alphabet_src_lut = {alphabet_src[key]: key for key in alphabet_src}
# alphabet_tgt_lut = {alphabet_tgt[key]: key for key in alphabet_tgt}

nsst = NSST.NSST_dict()
# iterate over pairs of sentences and alignment
with open(paired_language_file, 'r') as sentence_pairs:
    with open(alignment_file, 'r') as alignments:
        for sentence_pair, alignment_line in tqdm(zip(sentence_pairs, alignments),
                                                     desc="process sentence pairs"):
            sentence_src, sentence_tgt = sentence_pair[1:-2].split(" ||| ")
            alignment = [pair.split('-') for pair in alignment_line[:-1].split(" ")]
            tokens_src = [[alphabet_extended_src[word]] for word in sentence_src.split(" ") if len(word)]
            tokens_tgt = [[alphabet_tgt[word]] for word in sentence_tgt.split(" ") if len(word)]

            # get most likely state sequence of the source sentence & append a final state (-1)
            states_src = model.decode(tokens_src)[1]
            states_src = np.concatenate([states_src, [-1]])
            register_spans = ()

            for i in range(len(states_src) - 1):
                q = states_src[i]
                qn = states_src[i + 1]
                token = tokens_src[i][0]
                spans = NSST.span(alignment, i)

                register_operations = ""
                # no changes add identity rule
                if register_spans == spans:
                    for xi in range(len(spans)):
                        register_operations += f"x{xi}, "

                else:
                    reg_op = ""
                    for span in spans:
                        if sum([span == s for s in register_spans]) > 0:
                            register_operations += f"x{np.argmax([span == s for s in register_spans])}, "
                        else:
                            j = span[0]
                            while j <= span[1]:
                                j_is_start_of_span = [j == s[0] for s in register_spans]
                                if sum(j_is_start_of_span) > 0:
                                    cor_span = np.argmax(j_is_start_of_span)
                                    register_operations += f"x{cor_span} "
                                    j = register_spans[cor_span][1] + 1

                                else:
                                    register_operations += f"{tokens_tgt[j][0]} "
                                    j += 1
                            register_operations = register_operations[:-1] + ", "
                # print(register_operations)
                nsst.add_rule(current_state=q,
                              next_state=qn,
                              token=token,
                              register_operations=register_operations[:-2])

                register_spans = spans  # update current register_spans

nsst.save_rules("output/test")
