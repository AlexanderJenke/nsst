import os
from argparse import ArgumentParser

from tqdm import tqdm

parser = ArgumentParser(prog='python3 europarl_dataloader.py')
parser.add_argument("-src", "--source_lang", help="Cleaned source language file", required=True)
parser.add_argument("-tgt", "--target_lang", help="Cleaned target language file", required=True)
parser.add_argument("-o", "--output", help="output directory, default: output", default="output")
parser.add_argument("--tss", type=int, default=20,
                    help="Step size over train sentences, for n every n-th sentence is used. Default: 20")
args = parser.parse_args()

# take same part of basenames as name
name = ""
for s, t in zip(os.path.basename(args.source_lang), os.path.basename(args.target_lang)):
    if s == t:
        name += s
    else:
        break

output_file = os.path.join(args.output, f"{name}tss{args.tss}.paired")

# create file containing paired sentences for token alignment with fast_align
with open(output_file, 'w') as output:
    with open(args.source_lang, 'r')as src:
        with open(args.target_lang, 'r') as tgt:
            p = list(zip(src, tgt))[4096::args.tss]
            for s, t in tqdm(p):
                # empty lines with '.' to prevent alignment errors
                if s == " \n":
                    s = ". \n"
                if t == " \n":
                    t = ". \n"

                # save paired sentences separated by '|||' (required by fast_align)
                print(f" {s[:-1]}||| {t[:-1]}", file=output)
