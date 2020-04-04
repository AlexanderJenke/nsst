from tqdm import tqdm

import NSST

if __name__ == '__main__':
    nsst = NSST.NSST()
    nsst.load("output/nsst_tss20_th4_nSt54_Q0.pkl", doCheckRules=False)

    # src = input("Schreibe einen Satz:\n")
    src = "Frau Präsidentin !"  # "Sehr geehrte Damen und Herren , ich entschuldige mich vielmals ."
    import time

    s = time.time()

    token_src = [nsst.alphabet_src[word] if word in nsst.alphabet_src else 0 for word in src.split(" ") if len(word)]
    i = {(-1, ()): 1}  # (q, reg): prob
    o = {}

    for t in token_src:
        for q, reg in tqdm(i.keys(), desc=f"Apply rules to token '{t}'"):
            prob = i[(q, reg)]
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
                        o[(q_n,
                           reg_n)] += prob * r_prob
                    else:
                        o[(q_n, reg_n)] = prob * r_prob

        # output is new input
        # th = max(o.values()) * 0#.001  # reduce rules by probability
        # i, o = {k: o[k] for k in o if o[k] >= th}, {}
        i, o = {k: o[k] for k in sorted(o, key=lambda x: o[x], reverse=True)[:1000]}, {}  # only best 1000 states
        # i, o = o, {}

    # get best translation and print human readable
    sorted_res = sorted({k: i[k] for k in i
                         if len(k[1]) == 1  # only use results where there is on entry in the register left
                         },
                        key=lambda x: i[x],
                        reverse=True)  # sort by probability
    for res in sorted_res[:10]:  # print the
        print(f"{i[res] * 100 :.2f}%: ", end="")
        for w in res[1][0].split(' '):
            if len(w):
                print(nsst.alphabet_tgt_lut[int(w)], end=" ")
        print("")
    print(f"\nTranslated in {time.time() - s:.3f}s")
