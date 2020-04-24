import pickle
import re
from io import StringIO

import numpy as np
from tqdm import tqdm


class Rule:
    """ This class represents a rule.
    The Rule converges from one state to the next by reading the token and performing the register operations on the
    given register, resulting in a new register."""

    def __eq__(self, other: dict):
        """This function checks if the rule is equal to all properties listed in the given dict
        :param other: dict containing property: value to be checked
        :return: true if all properties are equal, else false
        """
        return np.asarray([self.__getattribute__(key) == other[key] for key in other], dtype=bool).all()

    def __str__(self):
        """This function returns a human readable representation of the rule.
        q{id of source state} -{id of token}-> [{register opperations}] (q{id of target state}) / {count of rule}

        The register operations for every target register are separated by a colon.
        The single operations within the register are separated by a space.
        A single operation consists of the id of the token to be added
            or a x followed by the id of the register to be copied.
            e.g.: x1 10 copies the first register and appends the token 10
        :return: human readable string
        """
        return f"q{self.current_state} -{self.token}-> [{self.register_operations}] (q{self.next_state}) / {self.count}"

    def __init__(self, current_state=None, next_state=None, token=None, register_operations="", count=1, line=None):
        """ Initialize the rule object.
        Should either contain current_state, next_state, token & register_operations or a parsable line
        :param current_state: source state the rule starts from
        :param next_state: target state the rule ends in
        :param token: token of the source language read by the rule
        :param register_operations: register operations performed to construct the target registers
        :param count: rule count (number of times the rule was created)
        :param line: parsable line (see __str__ representation) -> all parameters are parsed from the line
        """
        self.current_state: int = current_state
        self.next_state: int = next_state
        self.token: int = token
        self.register_operations: str = register_operations

        # split up the single operations of the register operations
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

        # get the number of required input registers (highest register copied by a register operation)
        reg_ids = re.findall(r'(?<=x)[0-9]+', self.register_operations)
        reg_ids.append('-1')  # default if no registers are used by the rule
        self.req_num_reg: int = max([int(r) for r in reg_ids]) + 1

    def parse_line(self, line):
        """This function parses a line in the format given by the __str__ function
        Sets current_state, next_state, token, register_operations & count according to given line
        :param line: parsable string representation
        """
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
        """Apply rule to a given register.
        This function assumes the current state is euqal to the current_state parameter.
        This has to be assured elsewhere!

        :param args: one argument per register.
        :return: tuple of resulting state and resulting registers
        """
        registers = ()  # init empty registers -> to be filles by register operations
        for reg in self._reg_op:  # for every register operation in the register operations
            register = ""  # init empty register
            for op in reg:  # for single operation in the register operation
                # copy given register if operations starts with 'x' (marking the copy register operation)
                if op[:1] == "x":
                    if len(args) > int(op[1:]) - 1:  # ensure the requested register is actually given, else ignore
                        register += f"{args[int(op[1:]) - 1]}"
                    else:
                        raise ReferenceError(
                            f"Tried to access register {int(op[1:]) - 1} but only registers 0-{len(args) - 1} exist!")

                # generate token according to the register operation (operation not starting with 'x')
                else:
                    register += f"{op} "

            registers += (register,)  # append register to target registers

        return self.next_state, registers

    def __radd__(self, other):
        """ This function privides the possibility to sum up the counts using the sum() function on a list of rules.
        :param other: other rule
        :rtype: int
        :return: sum of counts
        """
        return self.count + other


class NSST:
    """This NSST stores a set of rules accessible over the tuple (current_state, token, num_reg)
    """

    def __init__(self, tokenization_src={}, tokenization_tgt={}):
        """init the nsst by generating lookup tables (LUT) for the given tokenizations
        :param tokenization_src: source language tokenization
        :param tokenization_tgt: target language tokenization
        """
        self.tokenization_src = tokenization_src
        self.tokenization_tgt = tokenization_tgt
        self.tokenization_src_lut = {self.tokenization_src[key]: key for key in self.tokenization_src}
        self.tokenization_tgt_lut = {self.tokenization_tgt[key]: key for key in self.tokenization_tgt}
        self.rules = {}  # dict used to look up rules faster (get_rules function)
        self.all_rules = []  # list of all rules

    def load_rules(self, file, doCheckRules=False):
        """ Load rules provides in form of a human readable rule file
        :param file: file containing a rule in every line in representation according to __str__ function of rules
        :param doCheckRules: bool if the rule should be checked for existence in nsst resulting in addition of count
        """
        if not isinstance(file, StringIO):
            file = open(file, "r")

        for rule_line in tqdm(file, desc="loading rules"):
            rule = Rule(line=rule_line)  # parse line
            q = rule.current_state
            qn = rule.next_state
            t = rule.token
            reg_op = rule.register_operations
            num_reg = rule.req_num_reg

            # add new rule to nsst (not checking or not existing)
            if not doCheckRules or not sum(list(r == {'current_state': q,
                                                      'token': t,
                                                      'next_state': qn,
                                                      'register_operations': reg_op} for r in
                                                self.rules)):  # add new rule
                self.all_rules.append(rule)  # add to list of rules
                if (q, t, num_reg) not in self.rules:
                    self.rules[(q, t, num_reg)] = []
                self.rules[(q, t, num_reg)].append(rule)  # add to dict of rules

            # rule exists in nsst -> increase count
            else:
                [r for r in self.rules if r == {'current_state': q,
                                                'token': t,
                                                'next_state': qn,
                                                'register_operations': reg_op}][0] += rule.count
        file.close()

    def save_rules(self, file):
        """ Save all rules in human readable format to the given file
        :param file: file to save the rules to
        """
        if not isinstance(file, StringIO):
            file = open(file, "w")
        for rule in self.all_rules:
            print(rule, file=file)  # save rule to file
        if not isinstance(file, StringIO):
            file.close()

    def get_rules(self, q, qn, t, num_reg):
        """ Fast lookup of rules where parameters are equal to the given
            current_state, next_state, token & number of required registers.
            Using the dict of rules.
        :param q: current_state
        :param qn: next_state
        :param t: token
        :param num_reg: number of required registers
        :return: list of rules satisfying the conditions
        """
        return self.rules[(q, qn, t, num_reg)]

    def save(self, file):
        """ save the whole nsst to the given file
        :param file: file to save the nsst to
        :return:
        """
        rules_io = StringIO()
        self.save_rules(rules_io)  # generate human readable list of all rules
        rules = rules_io.getvalue()
        # combine the rules & tokenizations in a dict to be saved
        nsst_d = {
            'rules': rules,
            'tokenization_src': self.tokenization_src,
            'tokenization_tgt': self.tokenization_tgt,
        }
        with open(file, 'wb') as f:
            pickle.dump(nsst_d, f)  # save the dict

    def load(self, file, doCheckRules=False):
        """ Load a nsst from file
        :param file: file containing a pickled dict containing the tokenizations and rules
        :param doCheckRules: bool, check the rules for duplicates on loading
        """
        with open(file, 'rb') as f:
            nsst_d = pickle.load(f)  # read the file

        # load the rules
        rules_io = StringIO(nsst_d['rules'])
        self.load_rules(rules_io, doCheckRules=doCheckRules)

        # load the tokenizations and generate LUTs
        self.tokenization_src = nsst_d['tokenization_src']
        self.tokenization_tgt = nsst_d['tokenization_tgt']
        self.tokenization_src_lut = {self.tokenization_src[key]: key for key in self.tokenization_src}
        self.tokenization_tgt_lut = {self.tokenization_tgt[key]: key for key in self.tokenization_tgt}


class MinimalNSST(NSST):
    """ This NSST only stores the rule descriptions in a dict to save memory.
    Rules are generated on request by the 'get_rules' function.
    """

    def load_rules(self, file, doCheckRules=True):
        """ Load rules provides in form of a human readable rule file
        :param file: file containing a rule in every line in representation according to __str__ function of rules
        :param doCheckRules: this bool has no effect in this implementation of the nsst, as rules are always checked
                             for existence due to the storage architecture
        """
        if not doCheckRules:
            print("INFO: doCheckRules=False has no effect.")
        if not isinstance(file, StringIO):
            file = open(file, "r")  # load file
        for rule in tqdm(file, desc="loading rules"):
            r = Rule(line=rule)  # parse line
            self.add_rule(r.current_state, r.next_state, r.token, r.register_operations, r.count)  # add rule to nsst
        file.close()

    def save_rules(self, file):
        """ Save all rules in human readable format to the given file
        :param file: file to save the rules to
        """
        if not isinstance(file, StringIO):
            file = open(file, "w")
        for q in self.rules:
            for t in self.rules[q]:
                for qn, reg in self.rules[q][t]:
                    # save human readable representation of rule to file
                    print(f"q{q} -{t}-> [{reg}] (q{qn}) / {self.rules[q][t][(qn, reg)]}", file=file)
        if not isinstance(file, StringIO):
            file.close()

    def get_rules(self, q, t):
        """ Fast lookup of rules where parameters are equal to the given
            current_state & token.

        CAUTION: the required parameters differ from the super class!

        :param q: current_state
        :param t: token
        :return: list of rules satisfying the conditions
        """
        return [Rule(q, qn, t, reg, c) for (qn, reg), c in self.rules[q][t].items()]

    def add_rule(self, current_state, next_state, token, register_operations, count=1):
        """ Add a rule to the nsst or increase the count if it already exists.
        """
        # check if sorage architecture exists, else create
        if current_state not in self.rules:
            self.rules[current_state] = {}
        if token not in self.rules[current_state]:
            self.rules[current_state][token] = {}

        # add new rule
        if (next_state, register_operations) not in self.rules[current_state][token]:
            self.rules[current_state][token][(next_state, register_operations)] = count

        # increase count of existing rule
        else:
            self.rules[current_state][token][(next_state, register_operations)] += count


def span(alignment, src_pos):
    """This function calculates the covered span of a position in an alignment.
    :param alignment: the alignment
    :param src_pos: the id we are interested in
    :return: tuple of covered spans
    """
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
    """Debugging and Testing of the implemented functions"""

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
