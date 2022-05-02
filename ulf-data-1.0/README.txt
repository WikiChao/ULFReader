This directory has two dataset files:

ulf-1.0.json
  The dataset of the first official release of annotated ULFs.
  The dataset is a list of [id, sentence, ULF, ULF-AMR] entries.

ulf-1.0-stog.json
  A version of ulf-1.0.json with additional preprocessing steps to
  make it compatible with the sequence-to-graph AMR parser.

ann_doc_v1.pdf
  An anonymized version of version 1.0 of the ULF annotation guidelines that
  were used both to train annotators and as a reference during the annotation
  process.

In order to create the train/dev/test splits of the data please run the following command with the full dataset json file of interest.

python split-data.py --input [full dataset json file] --trainpath [train directory] --devpath [dev directory] --testpath [test directory]

