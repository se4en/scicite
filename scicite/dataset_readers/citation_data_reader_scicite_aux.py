from typing import Dict, List
import jsonlines
import logging

from overrides import overrides
from allennlp.common import Params
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, ELMoTokenCharactersIndexer

from scicite.helper import regex_find_citation

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("scicite_section_title_data_reader")
class SciciteSectitleDatasetReader(DatasetReader):

    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 clean_citation: bool = True,
                 with_elmo: bool = False
                 ) -> None:
        super().__init__(lazy)
        self._clean_citation = clean_citation
        self._tokenizer = tokenizer or SpacyTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        if with_elmo:
            self._token_indexers = {"elmo": ELMoTokenCharactersIndexer(),
                                    "tokens": SingleIdTokenIndexer()}
        else:
            self._token_indexers = {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        for obj in jsonlines.open(file_path):
            citation_text = obj['text']

            if self._clean_citation:
                citation_text = regex_find_citation.sub("", citation_text)

            citation_intent = None
            section_name = obj['section_name']
            citing_paper_id = obj['citing_paper_id']
            cited_paper_id = obj['cited_paper_id']

            yield self.text_to_instance(
                citation_text=citation_text,
                intent=citation_intent,
                citing_paper_id=citing_paper_id,
                cited_paper_id=cited_paper_id,
                section_name=section_name
            )

    @overrides
    def text_to_instance(self,
                         citation_text: str,
                         citing_paper_id: str,
                         cited_paper_id: str,
                         intent: List[str] = None,
                         section_name: str = None) -> Instance:

        citation_tokens = self._tokenizer.tokenize(citation_text)

        fields = {
            'citation_text': TextField(citation_tokens, self._token_indexers),
        }

        if section_name is not None:
            fields['section_label'] = LabelField(section_name, label_namespace="section_labels")

        fields['citing_paper_id'] = MetadataField(citing_paper_id)
        fields['cited_paper_id'] = MetadataField(cited_paper_id)
        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params):
        with_elmo = params.pop_bool("with_elmo", False)
        return cls(with_elmo=with_elmo)


@DatasetReader.register("scicite_cite_worthiness_data_reader")
class SciCiteWorthinessDataReader(DatasetReader):

    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 clean_citation: bool = True,
                 with_elmo: bool = False
                 # use_lexicon_features: bool = False,
                 # use_sparse_lexicon_features: bool = False
                 ) -> None:
        super().__init__(lazy)
        self._clean_citation = clean_citation
        self._tokenizer = tokenizer or SpacyTokenizer()
        if with_elmo:
            self._token_indexers = {"elmo": ELMoTokenCharactersIndexer(),
                                    "tokens": SingleIdTokenIndexer()}
        else:
            self._token_indexers = {"tokens": SingleIdTokenIndexer()}
        # if self.use_lexicon_features or self.use_sparse_lexicon_features:
        #     self.lexicons = {**ALL_ACTION_LEXICONS, **ALL_CONCEPT_LEXICONS}

    @overrides
    def _read(self, file_path):
        for obj in jsonlines.open(file_path):
            citation_text = obj['text']

            citation_intent = None
            section_name = None
            citing_paper_id = obj['citing_paper_id']
            cited_paper_id = obj['cited_paper_id']

            if self._clean_citation:
                citation_text = regex_find_citation.sub("", citation_text)

            yield self.text_to_instance(
                citation_text=citation_text,
                intent=citation_intent,
                citing_paper_id=citing_paper_id,
                cited_paper_id=cited_paper_id,
                cleaned_cite_text=obj['cleaned_cite_text'],
                is_citation=obj['is_citation']
            )

    @overrides
    def text_to_instance(self,
                         citation_text: str,
                         citing_paper_id: str,
                         cited_paper_id: str,
                         intent: List[str] = None,
                         cleaned_cite_text: str = None,
                         is_citation: bool = None) -> Instance:  # type: ignore

        citation_tokens = self._tokenizer.tokenize(citation_text)

        fields = {
            'citation_text': TextField(citation_tokens, self._token_indexers),
        }

        if is_citation is not None:
            fields['is_citation'] = LabelField(str(is_citation), label_namespace="cite_worthiness_labels")

        fields['citing_paper_id'] = MetadataField(citing_paper_id)
        fields['cited_paper_id'] = MetadataField(cited_paper_id)
        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params):
        with_elmo = params.pop_bool("with_elmo", False)
        return cls(with_elmo=with_elmo)
