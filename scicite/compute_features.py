""" Module for computing features """
import re
from collections import Counter, defaultdict
from typing import List, Optional, Tuple, Type, Dict

import functools
from spacy.tokens.token import Token as SpacyToken

import scicite.constants as constants
from scicite.constants import CITATION_TOKEN
from scicite.helper import find
from scicite.resources.lexicons import (AGENT_PATTERNS, ALL_ACTION_LEXICONS,
                                        ALL_CONCEPT_LEXICONS, FORMULAIC_PATTERNS)
from scicite.data import Citation
import logging

logger = logging.getLogger('classifier')

NETWORK_WEIGHTS_FILE = constants.root_path + '/resources/arc-network-weights.tsv'


def load_patterns(filename, p_dict, label):
    with open(filename) as f:
        class_counts = Counter()
        for line in f:
            if not '@' in line:
                continue
            cols = line.split("\t")
            pattern = cols[0].replace("-lrb-", "(").replace('-rrb-', ')')
            category = cols[1]
            if category == 'Background':
                continue
            class_counts[category] += 1
            p_dict[category + '_' + label + '_' + str(class_counts[category])] \
                = pattern.split()
            # p_dict[clazz + '_' + label].append(pattern.split())


# def is_in_patterns(patterns: dict,
#                   lexicon: dict,
#                   sentence: str,
#                   count: Optional[bool] = False) -> Tuple[List[str], List[str]]:
#     """ Checks if the patterns exist in the sentence """
#     pattern_features = []
#     pattern_names = []
#
#     for (feature_name, patterns) in patterns.iteritems():
#         if prefix is not None:
#             feature_name = prefix + feature_name
#         for pattern in patterns:
#             pat_index = find(pattern, citance, None, debug=debug, feature=feature_name)
#             if pat_index < 0:
#                 continue
#             # if debug is not None:
#             #     print('found %s in %s' % (pattern, debug))
#
#             # If the pattern happens after the citation
#             if cite_index < pat_index:
#                 offset = pat_index - cite_index
#             # Otherwise, it happens before, so take into account its length
#             else:
#                 offset = (pat_index - len(pattern)) - cite_index
#
#             features[feature_name] += 1
#             if CUR_LABEL is not None:
#                 FEATURE_FIRING_COUNTS[(' '.join(pattern), feature_name, CUR_LABEL.split("-")[0])] += 1
#
#
#     # if count:
#     #     cnt = 0
#     # for key, word_list in patterns.items():
#     #     exists = False
#
#         # for word in word_list:
#         #     if word in sentence:
#         #         if not count:
#         #             exists = True
#         #             break
#         #         else:
#         #             cnt += 1
#         # if not count:
#         #     features.append(exists)
#         # else:
#         #     features.append(cnt)
#         # feature_names.append(key)
#
#     return pattern_features, pattern_names


def get_formulaic_features(processed_sentence: List[Dict], count: bool = False, prefix=None):
    """
    Return furmulaic features from sentence
    """
    features = []
    feature_names = []
    pattern_names = []

    for (feature_name, patterns) in FORMULAIC_PATTERNS.items():
        if prefix is not None:
            feature_name = prefix + feature_name
        cnt = 0

        for pattern in patterns:
            prep_pat = pattern.split(' ')
            pat_index = find(prep_pat, processed_sentence, None, feature=feature_name)
            if pat_index >= 0:
                cnt += 1
                pattern_names.append(pattern)

        if count:
            features.append(cnt)
        else:
            features.append(cnt > 0)
        feature_names.append(feature_name)

    # Additionally, the 168 agent patterns are also considered as formulaic
    # patterns, wherever they do not occur as the subject of the sentence. The
    # decision to include these into the Formu feature was explained in section
    # 5.2.2.2.
    for (feature_name, patterns) in AGENT_PATTERNS.items():
        if prefix is not None:
            feature_name = prefix + feature_name
        cnt = 0

        for pattern in patterns:
            prep_pat = pattern.split(' ')
            pat_index = find(prep_pat, processed_sentence, False, feature=feature_name)
            if pat_index >= 0:
                cnt += 1
                pattern_names.append(pattern)

        if count:
            features.append(cnt)
        else:
            features.append(cnt > 0)
        feature_names.append(feature_name + ' (AS_FORM)')

    return features, feature_names, pattern_names


def get_agent_features(processed_sentence: List[Dict], count: bool = False, prefix=None):
    features = []
    feature_names = []
    pattern_names = []

    for (feature_name, patterns) in AGENT_PATTERNS.items():
        if prefix is not None:
            feature_name = prefix + feature_name
        cnt = 0

        for pattern in patterns:
            prep_pat = pattern.split(' ')
            pat_index = find(prep_pat, processed_sentence, True, feature=feature_name)
            if pat_index >= 0:
                cnt += 0
                pattern_names.append(pattern)

        if count:
            features.append(cnt)
        else:
            features.append(cnt > 0)
        feature_names.append(feature_name + ' (AS_AGENT)')
    return features, feature_names, pattern_names


def get_values_from_list(inplst, key, is_class=True):
    """ gets a value of an obj for a list of dicts (inplst)
    Args:
        inplst: list of objects
        key: key of interest
        is_class: ist the input object a class or a dictionary obj
    """
    return [getattr(elem, key) for elem in inplst] if is_class \
        else [elem[key] for elem in inplst]


def is_in_lexicon(lexicon: dict,
                  sentence: str,
                  count: Optional[bool] = False) -> Tuple[List[str], List[str]]:
    """ checks if the words in a lexicon exist in the sentence """
    features = []
    feature_names = []
    if count:
        cnt = 0
    for key, word_list in lexicon.items():
        exists = False
        for word in word_list:
            if word in sentence:
                if not count:
                    exists = True
                    break
                else:
                    cnt += 1
        if not count:
            features.append(exists)
        else:
            features.append(cnt)
        feature_names.append(key)
    return features, feature_names
