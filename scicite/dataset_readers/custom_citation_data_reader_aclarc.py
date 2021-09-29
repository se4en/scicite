from typing import Dict, List
import json
import jsonlines
import logging

import numpy as np
import torch
from transformers import AutoTokenizer
from allennlp.data import Field
from overrides import overrides
from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, ListField, ArrayField, MetadataField, Field
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, ELMoTokenCharactersIndexer

from scicite import AclarcDatasetReader
from scicite.resources.lexicons import ALL_ACTION_LEXICONS, ALL_CONCEPT_LEXICONS
from scicite.data import DataReaderJurgens
from scicite.data import read_jurgens_jsonline
from scicite.compute_features import is_in_lexicon, get_formulaic_features, get_agent_features

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

from scicite.constants import S2_CATEGORIES, NONE_LABEL_NAME


@DatasetReader.register("custom_aclarc_dataset_reader")
class CustomAclarcDatasetReader(AclarcDatasetReader):
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
        Tokenizer to use to split the title and abstrct into words or other kinds of tokens.
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
                 use_mask: bool = False,
                 with_bert: bool = False
                 ) -> None:
        super().__init__(lazy, tokenizer, token_indexers, use_lexicon_features,
                         use_sparse_lexicon_features, with_elmo)
        if with_bert:
            self.bert_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased", do_lower_case=True)
            self.bert_tokenizer.add_tokens("@@CITATION", special_tokens=True)
        self.use_mask = use_mask
        self.use_pattern_features = use_pattern_features

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
                sents_after=citation.sents_after,
                cleaned_cite_text=citation.cleaned_cite_text
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
                         sents_before: List[str] = None,
                         sents_after: List[str] = None,
                         cite_marker_begin: int = None,
                         cite_marker_end: int = None,
                         cleaned_cite_text: str = None,
                         citation_excerpt_index: str = None,
                         citation_id: str = None,
                         venue: str = None) -> Instance:  # type: ignore
        result = super().text_to_instance(citation_text, citing_paper_id, cited_paper_id, intent, citing_paper_title,
                                          cited_paper_title, citing_paper_year, cited_paper_year, citing_author_ids,
                                          cited_author_ids, extended_context, section_number, section_title,
                                          sents_before, sents_after, cite_marker_begin, cite_marker_end,
                                          cleaned_cite_text, citation_excerpt_index, citation_id, venue)
        # print("CIT_TEXT: ", citation_text, "CLEAN: ", cleaned_cite_text)
        result.fields['cit_text_for_bert'] = ArrayField(torch.Tensor(self.bert_tokenizer.encode(cleaned_cite_text
                                                                                                if self.use_mask
                                                                                                else citation_text,
                                                                                                padding='max_length',
                                                                                                max_length=400))
                                                        .to(torch.int32))

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

                # TODO: norm L2
            result.fields["pattern_features"] = ArrayField(torch.Tensor(formulaic_features + agent_features +
                                                                        formulaic_clause_features +
                                                                        agent_clause_features).to(torch.int32))
            #print("Patterns", result.fields['pattern_features'])
            #print("Embeddings", result.fields['cit_text_for_bert'])
                
        return result

    @classmethod
    def from_params(cls, params: Params) -> 'CustomAclarcDatasetReader':
        lazy = params.pop('lazy', False)
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        use_lexicon_features = params.pop_bool("use_lexicon_features", False)
        use_sparse_lexicon_features = params.pop_bool("use_sparse_lexicon_features", False)
        use_pattern_features = params.pop_bool("use_pattern_features", False)
        with_elmo = params.pop_bool("with_elmo", False)
        with_bert = params.pop_bool("with_bert", False)
        use_mask = params.pop_bool("use_mask", False)
        params.assert_empty(cls.__name__)
        return cls(lazy=lazy, tokenizer=tokenizer,
                   use_lexicon_features=use_lexicon_features,
                   use_sparse_lexicon_features=use_sparse_lexicon_features,
                   use_pattern_features=use_pattern_features,
                   with_elmo=with_elmo,
                   with_bert=with_bert,
                   use_mask=use_mask)
