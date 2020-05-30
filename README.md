# Nondeterministic Streaming String Transducer
![Datenfluss durch die einzelnen Verarbeitungsschritte.](Overview.png)


## Dependencies
- fast_align
- python >= 3.6
- hmmlearn >= 0.2.3
- scikit-learn
- numpy
- tensorboardX
- tqdm
- nltk >= 3.5
- pyter3

## File Structure

### Task relevant
- europarl_dataloader.py: implements functionality to preprocess europarl datasets
- HMM.py: implements a multi threaded MultinomialHMM
- hmm_training.py: implements the functionality to train the HMM
- alignment_createPairedFile.py: prepares data for the fast_align algorithm
- NSST.py: implements the NSST class to manage the extracted transitions and assignments
- nsst_createRules.py: implements the functionality to extract transitions and assignments
- output: folder contains all (interim) results of the single steps 
- runs: folder contains the tensorboard log files to monitor the HMM training
- docs: folder contains the project documentation
- requirements.txt: setup file for pip3


### Additional

- nsst_translate.py: implements the functionality to translate sentences using the NSST
- nsst_translate_corpus.py: translates the whole test corpus and saves it to a csv file
- nsst_score_csv.py: scores the csv file using the BLEU & TER Scores
- nsst_statistics.py: prints numerical statistics of the extracted NSSTs
- additional: folder contains further data created during the project but not relevant to the end result (mostly undocumented)

## Run
The shown commands use default parameters.
The parameters can be modified, for details run the python scripts with the ```-h``` parameter.

### Data Preprocessing 
To prepare the europarl data run the following 2 commands:
```python3
python3 europarl_dataloader.py -i output/europarl-v7-de-en.de
python3 europarl_dataloader.py -i output/europarl-v7-de-en.en
```
this will create the *europarl-v7-de-en.de.clean* und *europarl-v7-de-en.en.clean* files in the output directory.

### Hidden Marcov Model
To train a Multinomial HMM on the german corpus as source language run 
```python3
python3 hmm_training.py -i output/europarl-v7-de-en.de.clean
```
this will create the files *hmm_tss20_th4_nSt128_nIt100__[0-100].pkl*,  *hmm_tss20_th4_nSt128_nIt100.pkl* and *tokenization_tss20_th4.pkl* in the output directory as well as a tensorboard log in the runs directory.

The *hmm_..._[0-100].pkl* files contain interim states of the hmm.

The *hmm_...pkl* file contains the final hmm.

The *tokenization_tss20_th4.pkl* file contains the used tokenisation to map the source language to a machine readable alphabet.

### Alignment
The alignment extraction requires fast_align. 
Use the provided docker (see packages) or install fast_align from the [Repository](https://github.com/clab/fast_align).

To before extracting the alignments the sentences need to be prepared by running 
```python3
python3 alignment_createPairedFile.py -src output/europarl-v7-de-en.de.clean -tgt output/europarl-v7-de-en.en.clean
```
this will create the *europarl-v7.de-en.tss20.paired* file in the output directory.

To extract the alignments run 
```bash
fast_align -dov -N -i output/europarl-v7.de-en.tss20.paired > output/europarl-v7.de-en.tss20.align
```
this will create the *europarl-v7.de-en.tss20.align* file in the output directory containing the alignments.

### Nondeterministic Streaming String Transducer
To extract the NSST transitions and assignments run 
```python3
python3 nsst_createRules.py 
```
this will create the *nsst_tss20_th4_nSt200_Q0.pkl* file in the output directory containing the extracted transitions and assignments.

To print the extracted transitions and assignments run 
```python3
python3 nsst_printRules.py output/nsst_tss20_th4_nSt200_Q0.pkl
```