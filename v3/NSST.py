import re

import numpy as np


class Rule:
    current_state = None
    next_state = None
    token = None
    register_operations = None
    count = None

    def __eq__(self, other: dict):
        return np.asarray([self.__getattribute__(key) == other[key] for key in other], dtype=bool).all()
        # other['q'] == self.current_state & other['t'] == self.token

    def __str__(self):
        return f"q{self.current_state} -{self.token}-> [{self.register_operations}] (q{self.next_state}) / {self.count}"

    def __init__(self, current_state=None, next_state=None, token=None, register_operations=None, count=1, line=None):
        self.current_state = current_state
        self.next_state = next_state
        self.token = token
        self.register_operations = register_operations
        self.count = count
        if line is not None:
            self.parse_line(line)

    def parse_line(self, line):
        match = re.search(
            '^q(?P<current_state>[-0-9]+?) -(?P<token>[-0-9]+?)-> \[(?P<register_operations>[(\d)(x\d),\- ]+?)\] \(q(?P<next_state>[-0-9]+?)\) \/ (?P<count>[0-9]+?)$',
            line)

        self.current_state = int(match.group('current_state'))
        self.token = int(match.group('token'))
        self.next_state = int(match.group('next_state'))
        self.count = int(match.group('count'))
        self.register_operations = match.group('register_operations')

    def __call__(self, *args):
        registers = ()
        for reg in self.register_operations.split(', '):
            register = ""
            for op in reg.split(' '):
                if op.startswith("x"):
                    if len(args) > int(op[1:]) - 1:
                        register += f"{args[int(op[1:]) - 1]}"
                    else:
                        raise ReferenceError(
                            f"Tried to access register {int(op[1:]) - 1} but only registers 0-{len(args)-1} exist!")
                else:
                    register += f"{op} "
            registers += (register,)

        return self.next_state, registers

    def __radd__(self, other):
        return self.count + other


class NSST:
    """This NSST stores a set of rules in a list.

    """
    rules = []

    def load_rules(self, file):
        with open(file, "r") as rules:
            for rule in rules:
                self.rules.append(Rule(line=rule))

    def save_rules(self, file):
        with open(file, "w") as file:
            for rule in self.rules:
                print(rule, file=file)

    def get_rules(self, q, t):
        return list(filter(lambda x: x == {'current_state': q, 'token': t}, self.rules))

    def add_rule(self, current_state, next_state, token, register_operations):
        rule = list(filter(lambda x: x == {'current_state': current_state,
                                           'next_state': next_state,
                                           'token': token,
                                           'register_operations': register_operations},
                           self.rules))
        if len(rule):
            rule[0].count += 1

        else:
            self.rules.append(Rule(current_state=current_state,
                                   next_state=next_state,
                                   token=token,
                                   register_operations=register_operations,
                                   count=1))


class NSST_dict:
    """ This NSST only stores the rule descriptions in a dict to save memory.
    Rules are generated on request by the 'get_rules' function.
    """
    rules_d = {}

    def load_rules(self, file):
        with open(file, "r") as rules:
            for rule in rules:
                r = Rule(line=rule)
                self.add_rule(r.current_state, r.next_state, r.token, r.register_operations, r.count)

    def save_rules(self, file):
        with open(file, "w") as file:
            for q in self.rules_d:
                for t in self.rules_d[q]:
                    for qn, reg in self.rules_d[q][t]:
                        print(f"q{q} -{t}-> [{reg}] (q{qn}) / {self.rules_d[q][t][(qn, reg)]}", file=file)

    def get_rules(self, q, t):
        return [Rule(q, qn, t, reg, c) for (qn, reg), c in self.rules_d[q][t].items()]

    def add_rule(self, current_state, next_state, token, register_operations, count=1):
        if current_state not in self.rules_d:
            self.rules_d[current_state] = {}
        if token not in self.rules_d[current_state]:
            self.rules_d[current_state][token] = {}
        if (next_state, register_operations) not in self.rules_d[current_state][token]:
            self.rules_d[current_state][token][(next_state, register_operations)] = count
        else:
            self.rules_d[current_state][token][(next_state, register_operations)] += count


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

    exit()
    # example rules for:
    # src_sentence: 1 1 2
    # tgt_sentence: 1 1 3 2 2
    # alignment: 0-0 0-3 1-1 1-4 2-2
    # state sequence: 0 1 2 -1

    nsst = NSST_dict()
    nsst.load_rules("output/example_rules")
    # nsst.add_rule(0, 1, 1, "1, 2")
    # nsst.add_rule(1, 2, 1, "x1 1, x2 2")
    # nsst.add_rule(2, -1, 2, "x1 3 x2")

    # nsst = NSST()
    # nsst.load_rules("output/example_rules")

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
