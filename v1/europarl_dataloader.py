import os

EUROPARL_ROOT = "/Users/jenkealex/Documents/Uni/MA/VERT-2/P-PnS/europarl/output"
TOKEN_SPLIT_CHAR = ' '
INVALID_CHARS = '@#$,;:"`´\'\n'
INVALID_CHAR_MAP = {ord(c): None for c in INVALID_CHARS}


def scan_root():
    data_tree_d = {}
    data = os.listdir(EUROPARL_ROOT)
    for file in data:
        version, pair, language = file.split(".")
        if version not in data_tree_d:
            data_tree_d[version] = {}
        if pair not in data_tree_d[version]:
            data_tree_d[version][pair] = (pair.split('-')[0], pair.split('-')[1])

    return data_tree_d


def print_data_tree(data_tree_d: dict):
    for version in data_tree_d:
        print("{}".format(version))
        for pair in data_tree_d[version]:
            print("    {}".format(pair))


def select_dataset(data_tree_d: dict):
    version_l = list(data_tree_d)
    while True:
        print("Available Versions in directory '{}':".format(EUROPARL_ROOT))
        for i, v in enumerate(version_l):
            print("[{}]   {}".format(i, v))
        v_s = input("Enter [0-{}] to select europarl version:".format(len(version_l) - 1))
        if v_s.isdigit() and int(v_s) in range(len(version_l)):
            break
    v_i = int(v_s)

    pair_l = list(data_tree_d[version_l[v_i]])
    while True:
        print("Available language pairs in '{}':".format(version_l[v_i]))
        for i, p in enumerate(pair_l):
            print("[{}]   {}".format(i, p))
        p_s = input("Enter [0-{}] to select language pair:".format(len(pair_l) - 1))
        if v_s.isdigit() and int(p_s) in range(len(pair_l)):
            break
    p_i = int(p_s)

    lang_l = list(data_tree_d[version_l[v_i]][pair_l[p_i]])
    while True:
        print("Available languages in '{}.{}':".format(version_l[v_i], pair_l[p_i]))
        for i, p in enumerate(lang_l):
            print("[{}]   {}".format(i, p))
        l_s = input("Enter [0-{}] to select base language:".format(len(pair_l) - 1))
        if v_s.isdigit() and int(l_s) in range(len(lang_l)):
            break
    l_i = int(l_s)

    return "{}.{}.{}".format(version_l[v_i], pair_l[p_i], lang_l[l_i])


def get_corresponding_file(dataset: str):
    dataset_l = dataset.split(".")
    lang_l = dataset_l[1].split("-")
    lang_l.remove(dataset_l[2])
    return "{}.{}.{}".format(dataset_l[0], dataset_l[1], lang_l[0])


def clean_line(line: str):
    """ Cleans strings by:
        - removing characters listed in INVALID_CHARS
        - removing '- '
        - striping outer spaces
        - shrinking multiple spaces to a single space.
        - replacing "' s" with "s" e.g. "Mr Bean' s" -> "Mr Beans"
    :param line: String to be cleaned
    :return: cleaned string
    """
    return line.replace('\' s', 's').translate(INVALID_CHAR_MAP).replace('- ', '').replace('  ', ' ').strip()


def load_dataset(dataset):
    print("loading dataset ...")
    base_lang_file = os.path.join(EUROPARL_ROOT, dataset)
    pair_lang_file = os.path.join(EUROPARL_ROOT, get_corresponding_file(dataset))

    if not os.path.isfile(base_lang_file):
        raise FileNotFoundError("{} could not be found".format(base_lang_file))

    if not os.path.isfile(pair_lang_file):
        raise FileNotFoundError("{} could not be found".format(pair_lang_file))

    data_d = []

    # load base language
    with open(base_lang_file, 'r') as base_f:
        for line in base_f:
            data_d.append([clean_line(line).split(TOKEN_SPLIT_CHAR), None])

    # load corresponding language
    with open(pair_lang_file, 'r') as pair_f:
        for i, line in enumerate(pair_f):
            data_d[i][1] = clean_line(line).split(TOKEN_SPLIT_CHAR)

    # remove erroneous pairs (one language is missing the sentence)
    for i, e in enumerate(data_d):
        if e[0] == [''] or e[1] == ['']:
            print(i, e)
            print(data_d[i+1])
            print()
            data_d.remove(e)

    print(i, len(data_d), i-len(data_d))

    for i in range(100):
        print("{}\n{}\n".format(data_d[i][0], data_d[i][1]))

    print("done.")
    return data_d


if __name__ == '__main__':
    load_dataset("europarl-v7.fr-en.en")
