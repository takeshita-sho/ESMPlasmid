Traceback (most recent call last):
  File "/nfshomes/stakeshi/esm/esm/model/train.py", line 47, in <module>
    train_dataset = dataset.MetagenomeDataset(data_dir)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/nfshomes/stakeshi/esm/esm/model/dataset.py", line 51, in __init__
    tokenized_sequence = dna_alphabet.encode(seq)
                         ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/nfshomes/stakeshi/esm/esm/data.py", line 252, in encode
    return [self.tok_to_idx[tok] for tok in self.tokenize(text)]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/nfshomes/stakeshi/esm/esm/data.py", line 252, in <listcomp>
    return [self.tok_to_idx[tok] for tok in self.tokenize(text)]
            ~~~~~~~~~~~~~~~^^^^^
KeyError: 'Epoch'
