from tqdm import tqdm

# define filenames
source_lang = "output/europarl-v7.de-en.de.clean"
target_lang = "output/europarl-v7.de-en.en.clean"
output_file = "output/europarl-v7.de-en.paired"

# create file containing paired sentences for token alignment with fast_align
with open(output_file, 'w') as output:
    with open(source_lang, 'r')as src:
        with open(target_lang, 'r') as tgt:
            for s, t in tqdm(zip(src, tgt)):
                if s == " \n" or t == " \n":
                    continue
                print(f" {s[:-1]}||| {t[:-1]}", file=output)
