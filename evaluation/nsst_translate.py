import gc

from tqdm import tqdm

import NSST

if __name__ == '__main__':
    nsst = NSST.NSST()
    nsst.load("output/nsst_tss20_th4_nSt54_Q0.pkl", doCheckRules=False)

    t0 = [r for r in nsst.all_rules if r.token == 0]

    tokens = {}
    for rule in tqdm(nsst.all_rules):
        gen = tuple(set(x for x in rule.register_operations.replace(',', '').split(' ')
                        if len(x) and x[0] != 'x'))
        if rule.token not in tokens:
            tokens[rule.token] = set()

        tokens[rule.token].add(gen)

    import numpy as np

    print("#token", len(tokens))
    print("mean rules per token", np.mean([len(tokens[k]) for k in tokens]))
    print("#token 0", len(tokens[0]))
    del tokens[0]
    print("mean rules per token wo 0", np.mean([len(tokens[k]) for k in tokens]))

    exit()

    src = input("Schreibe einen Satz:\n")
    import time

    s = time.time()

    token_src = [nsst.alphabet_src[word] if word in nsst.alphabet_src else 0 for word in src.split(" ") if len(word)]
    # i = {(-1, ()): 1}  # (q0, reg): prob
    i = ((-1, (), 1),)
    # i = {(q, ()): 1 for q in set(k[0] for k in nsst.rules.keys())}
    o = {}

    for t in token_src:
        for q, reg, prob in tqdm(i,  # .keys(),
                                 desc=f"Apply rules to token '{t}'"):
            # prob = i[(q, reg)]
            # get all applicable rules (current state, token and required number of registers match)
            if (q, t, len(reg)) in nsst.rules:
                rules = nsst.rules[(q, t, len(reg))]
                # total count of rules to calculate probability of single rule
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
        # th = max(o.values()) * 0#.001  # reduce rules by probability
        # i, o = {k: o[k] for k in o if o[k] >= th}, {}
        # i, o = {k: o[k] for k in sorted(o, key=lambda x: o[x], reverse=True)[:1000]}, {}  # only best 1000 states
        i, o = tuple((k[0], k[1], o[k]) for k in o), {}
        gc.collect()

    # get best translations
    sorted_res = sorted((k for k in i
                         if len(k[1]) == 1  # only use results where there is on entry in the register left
                         # if k[0] == -1  # only use if in final state
                         ),
                        key=lambda x: x[2],  # sort by probability
                        reverse=True)  # highest prob first

    # print the best 10 results human readable
    for q, reg, prob in sorted_res[:10]:
        print(f"{prob * 100 :.2f}%: ", end="")
        for w in reg[0].split(' '):
            if len(w):
                print(nsst.alphabet_tgt_lut[int(w)], end=" ")
        print("")
    print(f"\nTranslated in {time.time() - s:.3f}s")
