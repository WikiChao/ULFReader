# ULFReader
An AllenNLP supported dataset loader for ULF dataset v1.0

    Dataset reader suitable for JSON-formatted ULF datasets.
    It will generate `Instances` with the following fields:
      - `ulf_words`, a `TextField`,
      - `tense`, a `SequenceLabelField`,
      - `class` another `SequenceLabelField`,
      - `multisent` a `LabelField`,
      - and `metadata`, a `MetadataField` that stores the instance's SID, the original sentence text,
        the original ULF annotations, amr annotations, parsed ULF annotations, accessible as `metadata['sid']`,
        `metadata['sentence']`, `metadata['raw_ulf']`, and
        `metadata['amr']`, and `metadata['parsed_ulf']`,respectively. This is so that we can more easily provide input to ULF models.
    We also support identifying the number of sentences, i.e., multisentetnce or not.
    We simply set the default value to words if they don't have any tense or class annotation.
    # Parameters
    tokenizer : `Tokenizer`, optional (default=`SpacyTokenizer()`)
        We use this `Tokenizer` for the ULF words.  See :class:`Tokenizer`.
        Default is `SpacyTokenizer()`.
    token_indexers : `Dict[str, TokenIndexer]`, optional
        We similarly use this for the ULF words.  See :class:`TokenIndexer`.
        Default is `{"tokens": SingleIdTokenIndexer()}`.
    multisent: `Optional[str]`, optional (default=`False`)
        A special token to append to each context. This is to help the sentence transition.
    
