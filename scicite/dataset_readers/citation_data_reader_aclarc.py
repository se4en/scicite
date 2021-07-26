""" Data reader for AllenNLP """

from typing import Dict, List
import json
import jsonlines
import logging

import torch
from allennlp.data import Field
from overrides import overrides
from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, MultiLabelField, ListField, ArrayField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, ELMoTokenCharactersIndexer

from scicite.scicite.resources.lexicons import ALL_ACTION_LEXICONS, ALL_CONCEPT_LEXICONS
from scicite.scicite.resources.lexicons import FORMULAIC_PATTERNS, AGENT_PATTERNS
from scicite.scicite.data import DataReaderJurgens
from scicite.scicite.data import read_jurgens_jsonline
from scicite.scicite.compute_features import is_in_lexicon, get_formulaic_features, get_agent_features
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

from scicite.scicite.constants import S2_CATEGORIES, NONE_LABEL_NAME


@DatasetReader.register("aclarc_dataset_reader")
class AclarcDatasetReader(DatasetReader):
    """
    Reads a JSON-lines file containing papers from the Semantic Scholar database, and creates a
    dataset suitable for document classification using these papers.

    where the ``label`` is derived from the citation intent

    Parameters
    ----------
    lazy : ``bool`` (optional, default=False)
        Passed to ``DatasetReader``.  If this is ``True``, training will start sooner, but will
        take longer per batch.  This also allows training with datasets that are too large to fit
        in memory.
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the title and abstract into words or other kinds of tokens.
        Defaults to ``WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input token representations. Defaults to ``{"tokens":
        SingleIdTokenIndexer()}``.
    """

    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 use_lexicon_features: bool = False,
                 use_sparse_lexicon_features: bool = False,
                 use_pattern_features: bool = False,
                 with_elmo: bool = False,
                 with_bert: bool = False
                 ) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        if with_bert:
            self.bert_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased", do_lower_case=True)
        if with_elmo:
            self._token_indexers = {"elmo": ELMoTokenCharactersIndexer(),
                                    "tokens": SingleIdTokenIndexer()}
        else:
            self._token_indexers = {"tokens": SingleIdTokenIndexer()}
        self.use_lexicon_features = use_lexicon_features
        self.use_sparse_lexicon_features = use_sparse_lexicon_features
        self.use_pattern_features = use_pattern_features
        if self.use_pattern_features:
            self.formulaic_patterns = FORMULAIC_PATTERNS
            self.agent_patterns = AGENT_PATTERNS
            self.lexicons = {**ALL_ACTION_LEXICONS, **ALL_CONCEPT_LEXICONS}
        elif self.use_lexicon_features or self.use_sparse_lexicon_features:
            self.lexicons = {**ALL_ACTION_LEXICONS, **ALL_CONCEPT_LEXICONS}

    @overrides
    def _read(self, file_path):
        for ex in jsonlines.open(file_path):
            citation = read_jurgens_jsonline(ex)
            yield self.text_to_instance(
                citation_text=citation.text,
                intent=citation.intent,
                citing_paper_id=citation.citing_paper_id,
                cited_paper_id=citation.cited_paper_id,
                sents_before=citation.sents_before,
                sents_after=citation.sents_after
            )

    @overrides
    def text_to_instance(self,
                         citation_text: str,
                         citing_paper_id: str,
                         cited_paper_id: str,
                         intent: List[str] = None,
                         citing_paper_title: str = None,
                         cited_paper_title: str = None,
                         citing_paper_year: int = None,
                         cited_paper_year: int = None,
                         citing_author_ids: List[str] = None,
                         cited_author_ids: List[str] = None,
                         extended_context: str = None,
                         section_number: int = None,
                         section_title: str = None,
                         sents_before: List[List] = None,
                         sents_after: List[List] = None,
                         cite_marker_begin: int = None,
                         cite_marker_end: int = None,
                         cleaned_cite_text: str = None,
                         citation_excerpt_index: str = None,
                         citation_id: str = None,
                         venue: str = None) -> Instance:  # type: ignore

        citation_tokens = self._tokenizer.tokenize(citation_text)
        # tok_cited_title = self._tokenizer.tokenize(cited_paper_title)
        # tok_citing_title = self._tokenizer.tokenize(citing_paper_title)
        # tok_extended_context = self._tokenizer.tokenize(extended_context)

        fields = {
            'citation_text': TextField(citation_tokens, self._token_indexers),
        }

        fields['cit_text_for_bert'] = self.bert_tokenizer.encode(citation_text, padding='max_length', max_length=300)

        if self.use_sparse_lexicon_features:
            # convert to regular string
            sent = [token.text.lower() for token in citation_tokens]
            lexicon_features, _ = is_in_lexicon(self.lexicons, sent)
            fields["lexicon_features"] = ListField([LabelField(feature, skip_indexing=True)
                                                    for feature in lexicon_features])

        if self.use_pattern_features:
            # sents_before[0] - citation sentence
            formulaic_features, _, _ = get_formulaic_features(sents_before[0], prefix='InCitSent:')
            agent_features, _, _ = get_agent_features(sents_before[0], prefix='InCitSent:')

            # compute patterns in clause
            formulaic_clause_features = formulaic_features
            agent_clause_features = agent_features
            if len(sents_before) > 1:
                for cur_sentence in sents_before[1:]:
                    _formulaic_features, _, _ = get_formulaic_features(cur_sentence, prefix='InClause:')
                    _agent_features, _, _ = get_agent_features(cur_sentence, prefix='InClause:')
                    formulaic_clause_features = [f_1 or f_2 for f_1, f_2 in zip(formulaic_clause_features,
                                                                                _formulaic_features)]
                    agent_clause_features = [f_1 or f_2 for f_1, f_2 in zip(agent_clause_features,
                                                                            _agent_features)]
            for cur_sentence in sents_after:
                _formulaic_features, _, _ = get_formulaic_features(cur_sentence, prefix='InClause:')
                _agent_features, _, _ = get_agent_features(cur_sentence, prefix='InClause:')
                formulaic_clause_features = [f_1 or f_2 for f_1, f_2 in zip(formulaic_clause_features,
                                                                            _formulaic_features)]
                agent_clause_features = [f_1 or f_2 for f_1, f_2 in zip(agent_clause_features,
                                                                        _agent_features)]

            fields["pattern_features"] = ListField([LabelField(feature, skip_indexing=True)
                                                    for feature in formulaic_features + agent_features +
                                                    formulaic_clause_features + agent_clause_features])

        if intent is not None:
            fields['labels'] = LabelField(intent)

        if citing_paper_year and cited_paper_year and \
                citing_paper_year > -1 and cited_paper_year > -1:
            year_diff = citing_paper_year - cited_paper_year
        else:
            year_diff = -1
        fields['year_diff'] = ArrayField(torch.Tensor([year_diff]))
        fields['citing_paper_id'] = MetadataField(citing_paper_id)
        fields['cited_paper_id'] = MetadataField(cited_paper_id)
        fields['citation_excerpt_index'] = MetadataField(citation_excerpt_index)
        fields['citation_id'] = MetadataField(citation_id)
        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params) -> 'AclarcDatasetReader':
        lazy = params.pop('lazy', False)
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        use_lexicon_features = params.pop_bool("use_lexicon_features", False)
        use_sparse_lexicon_features = params.pop_bool("use_sparse_lexicon_features", False)
        use_pattern_features = params.pop_bool("use_pattern_features", False)
        with_elmo = params.pop_bool("with_elmo", False)
        with_bert = params.pop_bool("with_bert", False)
        params.assert_empty(cls.__name__)
        return cls(lazy=lazy, tokenizer=tokenizer,
                   use_lexicon_features=use_lexicon_features,
                   use_sparse_lexicon_features=use_sparse_lexicon_features,
                   use_pattern_features=use_pattern_features,
                   with_elmo=with_elmo,
                   with_bert=with_bert)
