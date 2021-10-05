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

from scicite import SciciteDatasetReader
from scicite.resources.lexicons import ALL_ACTION_LEXICONS, ALL_CONCEPT_LEXICONS
from scicite.data import DataReaderJurgens
from scicite.data import DataReaderS2, DataReaderS2ExcerptJL
from scicite.data import read_jurgens_jsonline
from scicite.compute_features import is_in_lexicon, get_formulaic_features, get_agent_features

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

from scicite.constants import S2_CATEGORIES, NONE_LABEL_NAME


@DatasetReader.register("custom_scicite_dataset_reader")
class CustomSciciteDatasetReader(SciciteDatasetReader):
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
                 use_lexicon_features: bool = False,
                 use_sparse_lexicon_features: bool = False,
                 multilabel: bool = False,
                 use_pattern_features: bool = False,
                 with_elmo: bool = False,
                 use_mask: bool = False,
                 with_bert: bool = False,
                 reader_format: str = 'flat'
                 ) -> None:
        super().__init__(lazy, tokenizer, use_lexicon_features, multilabel,
                         use_sparse_lexicon_features, with_elmo, reader_format)
        if with_bert:
            self.bert_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased", do_lower_case=True)
            self.bert_tokenizer.add_tokens("@@CITATION", special_tokens=True)
        self.use_mask = use_mask
        self.use_pattern_features = use_pattern_features


    @overrides
    def _read(self, jsonl_file: str):
        if self.reader_format == 'flat':
            reader_s2 = DataReaderS2ExcerptJL(jsonl_file)
        elif self.reader_format == 'nested':
            # need use this
            reader_s2 = DataReaderS2(jsonl_file)
        for citation in reader_s2.read():
            yield self.text_to_instance(
                citation_text=citation.text,
                intent=citation.intent,
                citing_paper_id=citation.citing_paper_id,
                cited_paper_id=citation.cited_paper_id,
                citation_excerpt_index=citation.citation_excerpt_index,
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
                         venue: str = None) -> Instance:  # type: ignore
        result = super().text_to_instance(citation_text, citing_paper_id, cited_paper_id, intent, citing_paper_title,
                                          cited_paper_title, citing_paper_year, cited_paper_year, citing_author_ids,
                                          cited_author_ids, extended_context, section_number, section_title,
                                          cite_marker_begin, cite_marker_end, sents_before, sents_after,
                                          cleaned_cite_text, citation_excerpt_index, venue)

        result.fields['cit_text_for_bert'] = ArrayField(torch.Tensor(self.bert_tokenizer.encode(cleaned_cite_text
                                                                                                if self.use_mask
                                                                                                else citation_text,
                                                                                                padding='max_length',
                                                                                                max_length=400))
                                                        .to(torch.int32).cpu())

        if self.use_pattern_features:
            # sents_before[0] - citation sentence
            formulaic_features, _, _ = get_formulaic_features(sents_before[0], prefix='InCitSent:')
            agent_features, _, _ = get_agent_features(sents_before[0], prefix='InCitSent:')

            # TODO: norm L2
            result.fields["pattern_features"] = ArrayField(torch.Tensor(formulaic_features +
                                                                        agent_features).to(torch.int32))

        return result

    @classmethod
    def from_params(cls, params: Params) -> 'CustomAclarcDatasetReader':
        lazy = params.pop('lazy', False)
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        use_lexicon_features = params.pop_bool("use_lexicon_features", False)
        use_sparse_lexicon_features = params.pop_bool("use_sparse_lexicon_features", False)
        use_pattern_features = params.pop_bool("use_pattern_features", False)
        multilabel = params.pop_bool("multilabel")
        with_elmo = params.pop_bool("with_elmo", False)
        with_bert = params.pop_bool("with_bert", False)
        use_mask = params.pop_bool("use_mask", False)
        reader_format = params.pop("reader_format", 'nested')
        params.assert_empty(cls.__name__)
        return cls(lazy=lazy, tokenizer=tokenizer,
                   use_lexicon_features=use_lexicon_features,
                   use_sparse_lexicon_features=use_sparse_lexicon_features,
                   use_pattern_features=use_pattern_features,
                   multilabel=multilabel,
                   with_elmo=with_elmo,
                   with_bert=with_bert,
                   use_mask=use_mask,
                   reader_format=reader_format)
