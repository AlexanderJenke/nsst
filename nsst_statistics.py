from argparse import ArgumentParser

import numpy as np

from NSST import NSST

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--nsst", default="output/nsst_tss20_th4_nSt200_Q0.pkl")
    args = parser.parse_args()

    nsst = NSST()
    nsst.load(args.nsst, doCheckRules=False)

    n_rules = len(nsst.all_rules)
    n_rules_0 = len([r for r in nsst.all_rules if r.token == 0])

    generated_tokens = {}
    for rule in nsst.all_rules:
        if rule.token not in generated_tokens:
            generated_tokens[rule.token] = set()

        generated_tokens[rule.token].add(
            tuple(set(t for t in rule.register_operations.replace(',', '').split(' ') if len(t) and t[0] != 'x')))

    print(args.nsst)
    print(f"# Transitionen: {n_rules}")
    print(f"# Transitionen die Token 0 lesen: {n_rules_0} ({n_rules_0 / n_rules * 100:.2f}%)")
    print(f"Durchschnitt Verscheidene Token generiert durch den selben gelesenen Token: "
          f"{np.mean([len(v) for v in generated_tokens.values()])}+-"
          f"{np.std([len(v) for v in generated_tokens.values()])}")
    print(f"Durchschnitt Verscheidene Token generiert durch den selben gelesenen Token( ohne Token 0): "
          f"{np.mean([len(v) for k, v in generated_tokens.items() if k != 0])}+-"
          f"{np.std([len(v) for k, v in generated_tokens.items() if k != 0])}")
    print(f"mean count {np.mean([r.count for r in nsst.all_rules])}")
    print(f"mean count (ohne Token 0) {np.mean([r.count for r in nsst.all_rules if r.token != 0])}")
