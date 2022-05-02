import json
import logging
from typing import Any, Dict, List, Tuple, Optional, Iterable
import re
import argparse

from allennlp.common.file_utils import cached_path
from allennlp.data.fields import TextField, LabelField, SequenceLabelField
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, SpacyTokenizer
from allennlp.data.data_loaders import MultiProcessDataLoader
from allennlp_models.rc.dataset_readers import utils
from allennlp.data.fields import TextField, MetadataField
from allennlp.data.samplers import BatchSampler
from allennlp.data.vocabulary import Vocabulary

logger = logging.getLogger(__name__)

ULF_NO_TENSE_TOKEN = "@@<NO_TENSE>@@"
ULF_NO_CLASS_TOKEN = "@@<NO_CLASS>@@"

class ULFReader(DatasetReader):
    """
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
    """

    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
    ) -> None:

        super().__init__(
            manual_distributed_sharding=True, manual_multiprocess_sharding=True
        )
        self._tokenizer = tokenizer or SpacyTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.multisent = False

    def read(self, file_path: str):

        logger.info("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset_json = json.load(dataset_file)

        logger.info("Reading the dataset")
        for item in dataset_json:
            # read sid
            sid = item[0]

            # read raw sentence
            sentence = item[1]

            # read and parse ulf preprocess data
            ulf = item[2]
            parsed_ulf = re.split(r'\s*[()]\s*',ulf)
            if parsed_ulf[0] == 'MULTI-SENT':
                self.multisent = True
                parsed_ulf = parsed_ulf[1:]
            new_parsed_ulf = []
            for i in parsed_ulf:
                if i == '':
                    pass
                elif i.count('.') > 1 or 'TO' in i:
                    splited = i.split(' ')
                    for s in splited:
                        new_parsed_ulf.append(s)
                else:
                    new_parsed_ulf.append(i)

            ulf_word = []
            ulf_tense = []
            ulf_class = []

            # decompose ulf format
            for element in new_parsed_ulf:
                # get tense
                tense_group = element.split(' ')
                if len(tense_group) > 1:
                    ulf_tense.append(tense_group[0])
                    word_group = tense_group[1].split('.')
                else:
                    ulf_tense.append(ULF_NO_TENSE_TOKEN)
                    word_group = tense_group[0].split('.')

                # get word and class
                if len(word_group) > 1:
                    ulf_word.append(word_group[0])
                    ulf_class.append(word_group[1])
                else:
                    ulf_word.append(word_group[0])
                    ulf_class.append(ULF_NO_CLASS_TOKEN)

            # read amr structure data
            amr = item[3]

            instance = self.text_to_instance(
                ulf_word,
                ulf_tense,
                ulf_class,
                new_parsed_ulf,
                sid,
                sentence,
                ulf,
                amr,
            )
            if instance is not None:
                yield instance

    def text_to_instance(
        self,  # type: ignore
        ulf_word: List[str],
        ulf_tense: List[str],
        ulf_class: List[str],
        parsed_ulf,
        sid: str,
        sentence: str,
        ulf: str,
        amr: str,
    ) -> Optional[Instance]:
        for w in ulf_word:
            print(w)
        word_tokens = [Token(w) for w in ulf_word]
           
        text_field = TextField(word_tokens, self._token_indexers)
        fields: Dict[str, Field] = {"ulf_words": text_field}
        if ulf_tense:
            fields["tense"] = SequenceLabelField(ulf_tense,text_field)
        if ulf_class:
            fields["class"] = SequenceLabelField(ulf_class,text_field)
        fields["metadata"] = MetadataField({"sentence": sentence,
            'sid': sid, "raw_ulf":ulf, "amr": amr, "parsed_ulf": parsed_ulf})
        if self.multisent:
            fields["multisent"] = LabelField('True', label_namespace="multisentence")
        else:
            fields["multisent"] = LabelField('False', label_namespace="multisentence")
        return Instance(fields)


if __name__ == "__main__":

    parser = argparse.ArgumentParser('data loader')
    parser.add_argument('--input', default='./ulf-data-1.0/all.json', help='Input json file of data.')
    parser.add_argument('--batch_size', default=1, help='Number of example in a batch')
    args = parser.parse_args()

    with open(args.input) as dataset_file:
        dataset_json = json.load(dataset_file)

    reader = ULFReader()
    vocab = Vocabulary.from_instances(reader.read(args.input))

    print("Default:")
    data_loader = MultiProcessDataLoader(reader, args.input, batch_size=args.batch_size)
    data_loader.index_with(vocab)
    for batch in data_loader:
        print(batch)
