import pickle
import re
from io import StringIO

import numpy as np
from tqdm import tqdm


class Rule:
    def __eq__(self, other: dict):
        return np.asarray([self.__getattribute__(key) == other[key] for key in other], dtype=bool).all()
        # other['q'] == self.current_state & other['t'] == self.token

    def __str__(self):
        return f"q{self.current_state} -{self.token}-> [{self.register_operations}] (q{self.next_state}) / {self.count}"

    def __init__(self, current_state=None, next_state=None, token=None, register_operations="", count=1, line=None):
        self.current_state: int = current_state
        self.next_state: int = next_state
        self.token: int = token
        self.register_operations: str = register_operations

        self._reg_op = []
        for reg in self.register_operations.split(', '):
            if reg == '': continue
            h = []
            for op in reg.split(' '):
                if op != '':
                    h.append(op)
            self._reg_op.append(h)

        self.count: int = count
        if line is not None:
            self.parse_line(line)

        reg_ids = re.findall(r'(?<=x)[0-9]+', self.register_operations)
        reg_ids.append('-1')
        self.req_num_reg: int = max([int(r) for r in reg_ids]) + 1

    def parse_line(self, line):
        match = re.search(
            '^q(?P<current_state>[-0-9]+?) -(?P<token>[-0-9]+?)-> \[(?P<register_operations>[(\d)(x\d),\- ]*?)\] \(q(?P<next_state>[-0-9]+?)\) \/ (?P<count>[0-9]+?)$',
            line)

        self.current_state = int(match.group('current_state'))
        self.token = int(match.group('token'))
        self.next_state = int(match.group('next_state'))
        self.count = int(match.group('count'))
        self.register_operations = match.group('register_operations')
        self._reg_op = []
        for reg in self.register_operations.split(', '):
            if reg == '': continue
            h = []
            for op in reg.split(' '):
                if op != '':
                    h.append(op)
            self._reg_op.append(h)

    def __call__(self, *args):
        registers = ()
        for reg in self._reg_op:
            register = ""
            for op in reg:
                if op[:1] == "x":
                    if len(args) > int(op[1:]) - 1:
                        register += f"{args[int(op[1:]) - 1]}"
                    else:
                        raise ReferenceError(
                            f"Tried to access register {int(op[1:]) - 1} but only registers 0-{len(args) - 1} exist!")
                else:
                    register += f"{op} "
            registers += (register,)

        return self.next_state, registers

    def __radd__(self, other):
        return self.count + other


class NSST:
    """This NSST stores a set of rules accessible over the tuple (q,qn,t,num_reg)
    """

    def __init__(self, alphabet_src=[], alphabet_tgt=[]):
        self.alphabet_src = alphabet_src
        self.alphabet_tgt = alphabet_tgt
        self.alphabet_src_lut = {self.alphabet_src[key]: key for key in self.alphabet_src}
        self.alphabet_tgt_lut = {self.alphabet_tgt[key]: key for key in self.alphabet_tgt}
        self.rules = {}
        self.all_rules = []

    def load_rules(self, file, doCheckRules=True):
        if not isinstance(file, StringIO):
            file = open(file, "r")
        for rule_line in tqdm(file, desc="loading rules"):
            rule = Rule(line=rule_line)
            q = rule.current_state
            qn = rule.next_state
            t = rule.token
            reg_op = rule.register_operations
            num_reg = rule.req_num_reg

            if not doCheckRules or not sum(list(r == {'current_state': q,
                                                      'token': t,
                                                      'next_state': qn,
                                                      'register_operations': reg_op} for r in
                                                self.rules)):  # add new rule
                self.all_rules.append(rule)  # add to list of rules
                if (q, t, num_reg) not in self.rules:
                    self.rules[(q, t, num_reg)] = []
                self.rules[(q, t, num_reg)].append(rule)
            else:  # rule exists -> increase count
                [r for r in self.rules if r == {'current_state': q,
                                                'token': t,
                                                'next_state': qn,
                                                'register_operations': reg_op}][0] += rule.count
        file.close()

    def save_rules(self, file):
        if not isinstance(file, StringIO):
            file = open(file, "w")
        for rule in self.all_rules:
            print(rule, file=file)
        if not isinstance(file, StringIO):
            file.close()

    def get_rules(self, q, qn, t, num_reg):
        return self.rules[(q, qn, t, num_reg)]

    def save(self, file):
        rules_io = StringIO()
        self.save_rules(rules_io)
        rules = rules_io.getvalue()
        nsst_d = {
            'rules': rules,
            'alphabet_src': self.alphabet_src,
            'alphabet_tgt': self.alphabet_tgt,
        }
        with open(file, 'wb') as f:
            pickle.dump(nsst_d, f)

    def load(self, file, doCheckRules=False):
        with open(file, 'rb') as f:
            nsst_d = pickle.load(f)
        rules_io = StringIO(nsst_d['rules'])
        self.load_rules(rules_io, doCheckRules=doCheckRules)
        self.alphabet_src = nsst_d['alphabet_src']
        self.alphabet_tgt = nsst_d['alphabet_tgt']
        self.alphabet_src_lut = {self.alphabet_src[key]: key for key in self.alphabet_src}
        self.alphabet_tgt_lut = {self.alphabet_tgt[key]: key for key in self.alphabet_tgt}


class MinimalNSST(NSST):
    """ This NSST only stores the rule descriptions in a dict to save memory.
    Rules are generated on request by the 'get_rules' function.
    """

    def load_rules(self, file, doCheckRules=True):
        if not doCheckRules:
            print("INFO: doCheckRules=False has no effect.")
        if not isinstance(file, StringIO):
            file = open(file, "r")
        for rule in tqdm(file, desc="loading rules"):
            r = Rule(line=rule)
            self.add_rule(r.current_state, r.next_state, r.token, r.register_operations, r.count)
        file.close()

    def save_rules(self, file):
        if not isinstance(file, StringIO):
            file = open(file, "w")
        for q in self.rules:
            for t in self.rules[q]:
                for qn, reg in self.rules[q][t]:
                    print(f"q{q} -{t}-> [{reg}] (q{qn}) / {self.rules[q][t][(qn, reg)]}", file=file)
        if not isinstance(file, StringIO):
            file.close()

    def get_rules(self, q, t):
        return [Rule(q, qn, t, reg, c) for (qn, reg), c in self.rules[q][t].items()]

    def add_rule(self, current_state, next_state, token, register_operations, count=1):
        if current_state not in self.rules:
            self.rules[current_state] = {}
        if token not in self.rules[current_state]:
            self.rules[current_state][token] = {}
        if (next_state, register_operations) not in self.rules[current_state][token]:
            self.rules[current_state][token][(next_state, register_operations)] = count
        else:
            self.rules[current_state][token][(next_state, register_operations)] += count


def span(alignment, src_pos):
    pos = set([int(p[1]) for p in alignment if int(p[0]) <= src_pos])
    pos.add(-1)
    spans = ()
    a = None
    for i in range(max(pos) + 2):
        if i not in pos and a is not None:
            spans += (a, i - 1),
            a = None

        elif i in pos and a is None:
            a = i

    return spans


if __name__ == '__main__':
    alignment = [['0', '0'], ['1', '1'], ['3', '2'], ['5', '3'], ['6', '4'], ['4', '5'], ['4', '6'], ['7', '7'],
                 ['5', '8'], ['9', '9']]

    for i in range(10):
        print(i, set([int(p[1]) for p in alignment if int(p[0]) <= i]), span(alignment, i))

    # example rules for:
    # src_sentence: 1 1 2
    # tgt_sentence: 1 1 3 2 2
    # alignment: 0-0 0-3 1-1 1-4 2-2
    # state sequence: 0 1 2 -1

    nsst = MinimalNSST()
    nsst.load_rules("output/example_rules")
    # nsst.add_rule(0, 1, 1, "1, 2")
    # nsst.add_rule(1, 2, 1, "x1 1, x2 2")
    # nsst.add_rule(2, -1, 2, "x1 3 x2")

    # nsst = NSST()
    # nsst.load_rules("output/example_rules")

    print(nsst.rules)

    q = 0
    s = [1, 1, 2]
    c = 0
    args = ()
    for t in s:
        r = nsst.get_rules(q, t)[0]
        q, args = r(*args)
        print(q, args)
        if q == -1:
            print(*args)
            break
