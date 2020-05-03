import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import numpy as np
from tqdm import tqdm

import NSST
import europarl_dataloader as e_dl
from HMM import TbXMonitor

assert (TbXMonitor is not None)  # to keep import

create_rule_lock = Lock()  # create only one Rule at a time -> prevent race conditions on rule counter


def createRules_mostLikelyStateSequence(sentence_pair, alignment_line, hmm, nsst):
    """ This function creates translation rules for every token in  a given sentence pair, corresponding alignment
    and state sequence given by the HMM.
    :param sentence_pair: line containing source and target sentence separated by '|||'
    :param alignment_line: total alignment formatted according to fast_align, describing the sentence pair
    :param hmm: HMM object
    :param nsst: nsst object managing the rules
    """
    sentence_src, sentence_tgt = sentence_pair[1:-2].split(" ||| ")  # separate src and tgt sentence
    alignment = [pair.split('-') for pair in alignment_line[:-1].split(" ")]  # split alignment str into pieces

    # translate sentences to tokens
    tokens_src = [[nsst.tokenization_src[word]] for word in sentence_src.split(" ") if len(word)]
    tokens_tgt = [[nsst.tokenization_tgt[word]] for word in sentence_tgt.split(" ") if len(word)]

    # get most likely state sequence of the source sentence
    states_src = hmm.decode(tokens_src)[1]

    if doAppendQf:
        states_src = np.concatenate([states_src, [-1]])  # append final state qf := -1 -> q-q-q-qf
    else:
        states_src = np.concatenate([[-1], states_src])  # append start state q0 := -1 -> q0-q-q-q

    # initialize the given spans as empty
    current_spans = ()

    # create new rule for every state conversion
    for i in range(len(states_src) - 1):  # n states => n-1 state conversions
        q = states_src[i]  # current state
        qn = states_src[i + 1]  # next state
        token = tokens_src[i][0]  # token read by rule
        target_spans = NSST.span(alignment, i)  # spans covered after the rule is applied (goal to be created)

        register_operations = ""  # initialize register operations without any operation

        """In the following we build the target spans by combining given spans and adding tokens
        e.g.:
        given spans: x1: (0,4), x2: (6,7)
        target span x1: (0,7)
        resulting register_operation: x1 5 x2 (concatenating register x1, token 5 and register x2)
        
        if the rule generates multiple registers, the register operations per register are separated by a ','
        e.g.:
        given spans: x1: (0,4), x2: (6,7), x3: (10,11)
        target span x1: (0,7) x2: (10,11)
        resulting register_operation: x1 5 x2, x3  
        """

        for span in target_spans:  # create target spans out of existing spans and target sentence tokens
            csi = span[0]  # init currently searched index (csi) by the first index in target span
            while csi <= span[1]:  # if csi < the last index in target span
                # check if given span starts with csi
                csi_is_start_of_span = [csi == s[0] for s in current_spans]

                # a given span starts with the csi
                if sum(csi_is_start_of_span) > 0:
                    span_id = np.argmax(csi_is_start_of_span)  # id of the span
                    register_operations += f"x{span_id} "  # add this span to the register
                    csi = current_spans[span_id][1] + 1  # continue with next index after span

                # csi not in existing spans -> rule generates corresponding token
                else:
                    register_operations += f"{tokens_tgt[csi][0]} "  # add corresponding token to rule
                    csi += 1  # continue with next index

            register_operations = register_operations[:-1] + ", "  # finish register operation

        # add the rule 'q -token-> [register_operations](qn)' to the nsst
        with create_rule_lock:
            nsst.add_rule(current_state=q,
                          next_state=qn,
                          token=token,
                          register_operations=register_operations[:-2])

        # update current register_spans
        current_spans = target_spans


if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-mt", "--multi_threading", default=True)
    parser.add_argument("-qf", "--AppendQf", default=False,
                        help="extend state order with final state, else use initial state, default: initial state")
    parser.add_argument("-align", "--alignment_file", default="output/europarl-v7.de-en.tss20.align")
    parser.add_argument("-pair", "--paired_language_file", default="output/europarl-v7.de-en.tss20.paired")
    parser.add_argument("-hmm", "--hmm_file", default="output/hmm_tss20_th4_nSt200_nIt101.pkl")
    parser.add_argument("-tokenization", "--tokenization_file", default="output/tokenization_tss20_th4.pkl")
    parser.add_argument("-nsst", "--nsst_output_file", default="output/nsst_tss20_th4_nSt200_Q0.pkl")
    args = parser.parse_args()

    doAppendQf = args.AppendQf  # append final state? else append start state

    # load HMM
    with open(args.hmm_file, 'rb') as file:
        hmm = pickle.load(file)
    """ :type hmm: MultiThreadFit"""

    # load tokenization used with HMM
    with open(args.tokenization_file, 'rb') as file:
        tokenization_src = pickle.load(file)

    # create tokenization for target language & extend source tokenization if needed
    # (every word in the source language must maps to a token)
    with open(args.paired_language_file, 'r') as sentence_pairs:
        split_sentence_pairs = list(s[1:-2].split(" ||| ") for s in sentence_pairs)
        wordcount_src = e_dl.count_words((w for w in s[0].split(" ")) for s in split_sentence_pairs)
        wordcount_tgt = e_dl.count_words((w for w in s[1].split(" ")) for s in split_sentence_pairs)
        tokenization_tgt = e_dl.create_tokenization(wordcount_tgt)
        tokenization_extended_src = e_dl.extend_tokenization(tokenization_src, wordcount_src)

    # initialize a nsst object to manage the rules
    nsst = NSST.MinimalNSST(tokenization_extended_src, tokenization_tgt)

    # iterate over pairs of sentences and alignment
    with open(args.paired_language_file, 'r') as sentence_pairs:
        with open(args.alignment_file, 'r') as alignments:

            # run in paralell by creating a job per sentence (executed by threads)
            if args.multi_threading:
                with ThreadPoolExecutor() as executor:
                    # create jobs
                    jobs = [
                        executor.submit(createRules_mostLikelyStateSequence, sentence_pair, alignment_line, hmm, nsst)
                        for sentence_pair, alignment_line in tqdm(zip(sentence_pairs, alignments),
                                                                  desc="create job per sentence pair")]
                    # wait for the jobs to finish
                    for _ in tqdm(as_completed(jobs), total=len(jobs), desc="process jobs"):
                        pass

            # or run sequentially by calling the function for every sentence
            else:
                for sentence_pair, alignment_line in tqdm(zip(sentence_pairs, alignments),
                                                          desc="process sentence pairs"):
                    createRules_mostLikelyStateSequence(sentence_pair, alignment_line, hmm, nsst)

    # save the nsst (containing the generated rules)
    nsst.save(args.nsst_output_file)

    # save the generated rules human readable to a text file
    with open(args.nsst_output_file[:-3] + "txt") as file:
        nsst.save_rules(file)
