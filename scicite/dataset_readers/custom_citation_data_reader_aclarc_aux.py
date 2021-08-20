from typing import Dict, List
import jsonlines
import logging

import numpy as np
import torch
from overrides import overrides
from allennlp.common import Params
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, MultiLabelField, ListField, ArrayField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, ELMoTokenCharactersIndexer
from transformers import AutoTokenizer

from scicite import AclCiteWorthinessDatasetReader, AclSectionTitleDatasetReader
from scicite.helper import regex_find_citation

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("custom_aclarc_section_title_data_reader")
class CustomAclSectionTitleDatasetReader(AclSectionTitleDatasetReader):

    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 clean_citation: bool = True,
                 with_elmo: bool = False,
                 with_bert: bool = False
                 ) -> None:
        super().__init__(lazy, tokenizer, token_indexers, clean_citation, with_elmo)
        if with_bert:
            self.bert_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased", do_lower_case=True)

    @overrides
    def text_to_instance(self,
                         citation_text: str,
                         citing_paper_id: str,
                         cited_paper_id: str,
                         intent: List[str] = None,
                         venue: str = None,
                         section_name: str = None) -> Instance:
        result = super().text_to_instance(citation_text, citing_paper_id, cited_paper_id, intent, venue, section_name)
        result.fields['cit_text_for_bert'] = ArrayField(torch.Tensor(self.bert_tokenizer.encode(citation_text,
                                                                                                padding='max_length',
                                                                                                max_length=400))
                                                        .to(torch.int32).cpu())
        return result

    @classmethod
    def from_params(cls, params: Params) -> 'CustomAclSectionTitleDatasetReader':
        lazy = params.pop('lazy', False)
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        with_elmo = params.pop_bool("with_elmo", False)
        with_bert = params.pop_bool("with_bert", False)
        params.assert_empty(cls.__name__)
        return cls(lazy=lazy, tokenizer=tokenizer,
                   with_elmo=with_elmo, with_bert=with_bert)


@DatasetReader.register("custom_aclarc_cite_worthiness_data_reader")
class CustomAclCiteWorthinessDatasetReader(AclCiteWorthinessDatasetReader):

    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 clean_citation: bool = True,
                 with_elmo: bool = False,
                 with_bert: bool = False
                 ) -> None:
        super().__init__(lazy, tokenizer, token_indexers, clean_citation, with_elmo)
        if with_bert:
            self.bert_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased", do_lower_case=True)

    @overrides
    def text_to_instance(self,
                         citation_text: str,
                         citing_paper_id: str,
                         cited_paper_id: str,
                         intent: List[str] = None,
                         cleaned_cite_text: str = None,
                         section_name: str = None,
                         is_citation: bool = None) -> Instance:
        result = super().text_to_instance(citation_text, citing_paper_id, cited_paper_id, intent, cleaned_cite_text,
                                          section_name, is_citation)
        result.fields['cit_text_for_bert'] = ArrayField(torch.Tensor(self.bert_tokenizer.encode(citation_text,
                                                                                                padding='max_length',
                                                                                                max_length=400))
                                                        .to(torch.int32).cpu())
        return result

    @classmethod
    def from_params(cls, params: Params) -> 'CustomAclCiteWorthinessDatasetReader':
        lazy = params.pop('lazy', False)
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        with_elmo = params.pop_bool("with_elmo", False)
        with_bert = params.pop_bool("with_bert", False)
        params.assert_empty(cls.__name__)
        return cls(lazy=lazy, tokenizer=tokenizer, with_elmo=with_elmo, with_bert=with_bert)
