import pickle
import time
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm

import NSST
from HMM import TbXMonitor

assert (TbXMonitor is not None)  # to keep import


def brute_force(args):
    """ try all possible rule sequences """

    # read input sentence, if not given in args
    if args.input is None:
        src = input("Schreibe einen Satz:\n")
    else:
        src = args.input

    start_time = time.time()

    # create tokenization for input sentence
    token_src = [nsst.tokenization_src[word] if word in nsst.tokenization_src else 0 for word in src.split(" ") if
                 len(word)]

    # initialize Register
    i = ((-1, (), 1),)  # ((initial state q0, register, prob),)
    o = {}

    # iterate over input sentence tokens
    for t in token_src:

        # iterate over possible nsst states
        for q, reg, prob in tqdm(i, desc=f"Apply rules to token '{t}'"):

            # get all applicable rules (current state, token and [required number of registers match])
            rule_descriptor = (q, t, len(reg)) if args.enforce_n_reg else (q, t)
            if rule_descriptor in nsst.rules:

                # all applicable rules (current state, input token & number of required registers match)
                rules = nsst.rules[rule_descriptor]

                # calculate total count of all applicable rules to later calculate the probability of a single rule
                sum_rules = sum(rules)

                # apply every rule
                for rule in rules:
                    q_n, reg_n = rule(*reg)  # apply rule
                    r_prob = rule.count / sum_rules  # rule probability = rule count / all counts

                    # if the new translation state was already reached via a different rule, add up the probabilities
                    if (q_n, reg_n) in o:
                        o[(q_n, reg_n)] += prob * r_prob
                    else:
                        o[(q_n, reg_n)] = prob * r_prob

        # output is new input
        i, o = tuple((k[0], k[1], o[k]) for k in o), {}
    print(f"\nTranslated in {time.time() - start_time:.3f}s")

    return i


def best_rule(args):
    """ only use the best rule per token """

    # read input sentence, if not given in args
    if args.input is None:
        src = input("Schreibe einen Satz:\n")
    else:
        src = args.input

    start_time = time.time()

    # create tokenization for input sentence
    token_src = [nsst.tokenization_src[word] if word in nsst.tokenization_src else 0 for word in src.split(" ") if
                 len(word)]

    # initialize Register
    i = (-1, (), 1)  # ((initial state q0, register, prob),)

    # iterate over input sentence tokens
    for t in tqdm(token_src, desc=f"Translating"):
        q, reg, prob = i

        # get all applicable rules (current state, token and [required number of registers match])
        rule_descriptor = (q, t, len(reg)) if args.enforce_n_reg else (q, t)
        if rule_descriptor in nsst.rules:

            # all applicable rules (current state, input token & number of required registers match)
            rules = nsst.rules[rule_descriptor]

            # calculate total count of all applicable rules to later calculate the probability of a single rule
            sum_rules = sum(rules)

            rule = sorted(rules, key=lambda x: x.count, reverse=True)[0]  # select best rule

            q_n, reg_n = rule(*reg)  # apply rule
            r_prob = rule.count / sum_rules  # rule probability = rule count / all counts

            i = (q_n, reg_n, prob * r_prob)  # update current state, register & probability
        else:
            raise RuntimeError(f"Found no applicable rule for q={q}, reg={reg}, t={t}!")

    print(f"\nTranslated in {time.time() - start_time:.3f}s")

    return (i,)


def best_transition_sequence(args):
    """ beste Transitionsreihenfolge """

    # read input sentence, if not given in args
    if args.input is None:
        src = input("Schreibe einen Satz:\n")
    else:
        src = args.input

    start_time = time.time()

    # create tokenization for input sentence
    token_src = [nsst.tokenization_src[word] if word in nsst.tokenization_src else 0 for word in src.split(" ") if
                 len(word)]

    # initialize Register
    i = ((-1, (), 1),)  # ((initial state q0, register, prob),)
    o = {}

    # iterate over input sentence tokens
    for t in token_src:

        # iterate over possible nsst states
        for q, reg, prob in tqdm(i, desc=f"Apply rules to token '{t}'"):

            # get all applicable rules (current state, token and [required number of registers match])
            rule_descriptor = (q, t, len(reg)) if args.enforce_n_reg else (q, t)
            if rule_descriptor in nsst.rules:

                # all applicable rules (current state, input token & number of required registers match)
                rules = nsst.rules[rule_descriptor]

                # calculate total count of all applicable rules to later calculate the probability of a single rule
                sum_rules = sum(rules)

                # apply every rule
                for rule in rules:
                    q_n, reg_n = rule(*reg)  # apply rule
                    r_prob = rule.count / sum_rules  # rule probability = rule count / all counts

                    # if the new translation state was already reached via a different rule, add up the probabilities
                    n_prob = prob * r_prob
                    if (q_n not in o) or (q_n in o and n_prob > o[q_n][1]):
                        o[q_n] = (reg_n, n_prob)  # use if state was never reached before or this path is better

        # output is new input
        i, o = tuple((k, o[k][0], o[k][1]) for k in o), {}
    print(f"\nTranslated in {time.time() - start_time:.3f}s")

    return i


def hmm(args):
    """ hmm provides most likely state sequence, only use applicable rules """

    # load HMM
    with open(args.hmm, 'rb') as file:
        hmm_model = pickle.load(file)
    """ :type hmm_model: MultiThreadFit"""

    # read input sentence, if not given in args
    if args.input is None:
        src = input("Schreibe einen Satz:\n")
    else:
        src = args.input

    start_time = time.time()

    # create tokenization for input sentence
    token_src = [[nsst.tokenization_src[word]] if word in nsst.tokenization_src else [0] for word in src.split(" ") if
                 len(word)]

    # hmm calculates the best state sequence
    states_src = []
    if 'Q0' in args.nsst:
        states_src = np.concatenate([[-1], hmm_model.decode(token_src)[1]])  # append start state q0 := -1 -> q0-q-q-q
    elif 'Qf' in args.nsst:
        states_src = np.concatenate([hmm_model.decode(token_src)[1], [-1]])  # append final state qf := -1 -> q-q-q-qf

    # initialize Register
    i = (((), 1),)  # ((register, prob),)
    o = {}

    # iterate over tuple of input sentence tokens, current state & next state
    for token, q, qn in zip(token_src, states_src[:-1], states_src[1:]):
        t = token[0]

        # iterate over possible nsst states
        for reg, prob in tqdm(i, desc=f"Apply rules to token '{t}'"):

            # get all applicable rules (current state, token and required number of registers match)
            rule_descriptor = (q, t, len(reg)) if args.enforce_n_reg else (q, t)
            if rule_descriptor in nsst.rules:  # (q, t, len(reg))

                # all applicable rules (current state, next state, input token [& number of required registers match])
                rules = [rule for rule in nsst.rules[rule_descriptor]
                         if rule.next_state == qn]  # only allow rules ending up in the right next state

                # calculate total count of all applicable rules to later calculate the probability of a single rule
                sum_rules = sum(rules)

                # apply every rule
                for rule in rules:
                    _, reg_n = rule(*reg)  # apply rule
                    r_prob = rule.count / sum_rules  # rule probability = rule count / all counts

                    # if the new translation state was already reached via a different rule, add up the probabilities
                    if reg_n in o:
                        o[reg_n] += prob * r_prob
                    else:
                        o[reg_n] = prob * r_prob

        # output is new input
        i, o = tuple((k, p) for k, p in o.items()), {}
    print(f"\nTranslated in {time.time() - start_time:.3f}s")

    return tuple((qn, k, p) for k, p in i)


if __name__ == '__main__':
    # read parameters
    parser = ArgumentParser()
    parser.add_argument("--nsst", default="output/nsst_tss20_th4_nSt200_Q0.pkl", help="nsst file")
    parser.add_argument("--hmm", default="output/hmm_tss20_th4_nSt200_nIt101.pkl",
                        help="hmm file (required for mode=hmm)")
    parser.add_argument("--mode",
                        default="best_transition_sequence",
                        help="translation mode: ['brute_force', 'hmm', 'best_transition_sequence', 'best_rule']\n"
                             "    brute_force: try all possible rule sequences (very slow and high memory usage)\n"
                             "    hmm: hmm provides most likely state sequence, only use applicable rules (translation probability incorrect)\n"
                             "    best_rule: only use the best rule per token\n"
                             "    best_transition_sequence: only use best path per state (default)")
    parser.add_argument("--enforce_n_reg", default=False, help="Only use rules that use only but all the given registers, default: False")
    parser.add_argument("--enforce_n_final_reg", default=False, help="When appending start state only accept results having one string variable in register left, default:False")
    parser.add_argument("-i", "--input", default=None)
    args = parser.parse_args()

    # load NSST
    nsst = NSST.NSST()
    nsst.load(args.nsst, doCheckRules=False)

    # run translation
    result = None
    print(f"mode: {args.mode}")
    if args.mode == "brute_force":
        result = brute_force(args)
    elif args.mode == "hmm":
        result = hmm(args)
    elif args.mode == "best_rule":
        result = best_rule(args)
    elif args.mode == "best_transition_sequence":
        result = best_transition_sequence(args)
    else:
        print("UNKNOWN MODE!")
        parser.print_help()
        exit()

    # get best translations
    print(f"{len(result)} verschiedene Endzustände")
    sorted_res = sorted((k for k in result
                         # only use results where there is on entry in the register left (added q0)
                         if ('Qf' in args.nsst or not args.enforce_n_final_reg or len(k[1]) == 1)
                         # only use results if in final state (added qf)
                         and ('Q0' in args.nsst or k[0] == -1)
                         ),
                        key=lambda x: x[2],  # sort by probability
                        reverse=True)  # highest prob first

    # print the best 10 results human readable
    print(f"{len(sorted_res)} gültige Ergebnisse\n"
          f"Top 10:")
    for q, reg, prob in sorted_res[:10]:
        print(f"    {prob * 100 :.2f}%: ", end="")
        for w in reg[0].split(' '):
            if len(w):
                print(nsst.tokenization_tgt_lut[int(w)], end=" ")
        print("")
