from tqdm import tqdm

# define filenames
source_lang = "output/europarl-v7.de-en.de.clean"
target_lang = "output/europarl-v7.de-en.en.clean"
output_file = "output/europarl-v7.de-en.tss20.paired"

# create file containing paired sentences for token alignment with fast_align
with open(output_file, 'w') as output:
    with open(source_lang, 'r')as src:
        with open(target_lang, 'r') as tgt:
            p = list(zip(src, tgt))[4096::20]
            for s, t in tqdm(p):
                if s == " \n":
                    s = ". \n"
                if t == " \n":
                    t = ". \n"
                print(f" {s[:-1]}||| {t[:-1]}", file=output)
