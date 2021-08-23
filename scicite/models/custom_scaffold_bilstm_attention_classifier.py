import operator
from copy import deepcopy
from distutils.version import StrictVersion
from typing import Dict, Optional

import allennlp
import numpy as np
import torch
import torch.nn.functional as F
from allennlp.common import Params, Registrable
from allennlp.data import Instance
from allennlp.data import Vocabulary
from allennlp.data.dataset import Batch
from allennlp.data.fields import TextField, LabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2VecEncoder, Seq2SeqEncoder, TextFieldEmbedder, Embedding, TimeDistributed
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from overrides import overrides
from torch.nn import Parameter, Linear
from transformers import AutoModel

from scicite import ScaffoldBilstmAttentionClassifier
from scicite.constants import Scicite_Format_Nested_Jsonlines

import torch.nn as nn

from scicite.models.scaffold_bilstm_attention_classifier import Attention


@Model.register("custom_scaffold_bilstm_attention_classifier")
class CustomScaffoldBilstmAttentionClassifier(ScaffoldBilstmAttentionClassifier):
    """
    This ``Model`` performs text classification for citation intents.  We assume we're given a
    citation text, and we predict some output label.
    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 citation_text_encoder: Seq2SeqEncoder,
                 classifier_feedforward: FeedForward,
                 classifier_feedforward_2: FeedForward,
                 classifier_feedforward_3: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 report_auxiliary_metrics: bool = False,
                 predict_mode: bool = False,
                 weighted_loss: bool = False,
                 focal_loss: bool = False,
                 use_mask: bool = False,
                 use_cnn: bool = False,
                 tokenizer_len: int = 31090,
                 bert_model: Optional[AutoModel] = None,
                 ) -> None:
        """
        Additional Args:
            lexicon_embedder_params: parameters for the lexicon attention model
            use_sparse_lexicon_features: whether to use sparse (onehot) lexicon features
            multilabel: whether the classification is multi-label
            data_format: s2 or jurgens
            report_auxiliary_metrics: report metrics for aux tasks
            predict_mode: predict unlabeled examples
            weighted_loss: use weighted CE
            focal_loss: use focal loss
        """
        super().__init__(vocab, text_field_embedder, citation_text_encoder, classifier_feedforward,
                         classifier_feedforward_2, classifier_feedforward_3, initializer, regularizer,
                         report_auxiliary_metrics, predict_mode)
        self.bert_model = bert_model
        self.use_mask = use_mask
        self.use_cnn = use_cnn
        if self.bert_model is not None:
            for param in self.bert_model.parameters():
                param.requires_grad = False
            if self.use_mask:
                self.bert_model.resize_token_embeddings(tokenizer_len)
        self.weighted_loss = weighted_loss
        self.focal_loss = focal_loss

        if self.weighted_loss:
            weights = [0.32447342, 0.88873626, 0.92165242, 3.67613636, 4.49305556, 4.6884058]
            class_weights = torch.FloatTensor(weights)  # .cuda()
            self.loss_main_task = torch.nn.CrossEntropyLoss(weight=class_weights)
        elif self.focal_loss:
            self.loss_main_task = torch.hub.load(
                'adeelh/pytorch-multi-class-focal-loss',
                model='FocalLoss',
                alpha=torch.tensor([0.05, 0.1, 0.1, 0.24, 0.25, 0.26]),
                gamma=2,
                reduction='mean',
                force_reload=False
            )

        initializer(self)

    @overrides
    def forward(self,
                citation_text: Dict[str, torch.LongTensor],
                labels: torch.LongTensor = None,
                lexicon_features: Optional[torch.IntTensor] = None,
                pattern_features: Optional[torch.IntTensor] = None,
                year_diff: Optional[torch.Tensor] = None,
                citing_paper_id: Optional[str] = None,
                cited_paper_id: Optional[str] = None,
                citation_excerpt_index: Optional[str] = None,
                citation_id: Optional[str] = None,
                section_label: Optional[torch.Tensor] = None,
                is_citation: Optional[torch.Tensor] = None,
                cit_text_for_bert: Optional[torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model
        Args:
            citation_text: citation text of shape (batch, sent_len, embedding_dim)
            labels: labels
            lexicon_features: lexicon sparse features (batch, lexicon_feature_len)
            pattern_features: pattern sparse features (batch, pattern_feature_len)
            year_diff: difference between cited and citing years
            citing_paper_id: id of the citing paper
            cited_paper_id: id of the cited paper
            citation_excerpt_index: index of the excerpt
            citation_id: unique id of the citation
            section_label: label of the section
            is_citation: citation worthiness label
            cit_text_for_bert: encode citation text for bert model
        """
        # pylint: disable=arguments-differ
        if self.bert_model is not None:
            new_cit_tex_for_bert = cit_text_for_bert.to(torch.int32).cpu()
            citation_text_embedding = self.bert_model(new_cit_tex_for_bert, return_dict=False)[0]
            citation_text_embedding = citation_text_embedding.to(torch.int32).cuda()
            citation_text_mask = (cit_text_for_bert == 0).to(torch.int32).cuda()
            # TODO look on paddings
            encoded_citation_text = self.citation_text_encoder(citation_text_embedding, citation_text_mask)
        else:
            citation_text_embedding = self.text_field_embedder(citation_text)
            citation_text_mask = util.get_text_field_mask(citation_text)

            # shape: [batch, sent, output_dim]
            encoded_citation_text = self.citation_text_encoder(citation_text_embedding, citation_text_mask)

        if not self.use_cnn:
            # shape: [batch, output_dim]
            attn_dist, encoded_citation_text = self.attention_seq2seq(encoded_citation_text, return_attn_distribution=True)

        # In training mode, labels are the citation intents
        # If in predict_mode, predict the citation intents
        if labels is not None:
            if pattern_features is not None:
                feedforward_input = torch.cat((pattern_features, encoded_citation_text), 1)
                logits = self.classifier_feedforward(feedforward_input)
            else:
                logits = self.classifier_feedforward(encoded_citation_text)

            class_probs = F.softmax(logits, dim=1)

            output_dict = {"logits": logits}

            if self.focal_loss or self.weighted_loss:
                loss = self.loss_main_task(logits, labels)
            else:
                loss = self.loss(logits, labels)

            output_dict["loss"] = loss

            # compute F1 per label
            for i in range(self.num_classes):
                metric = self.label_f1_metrics[self.vocab.get_token_from_index(index=i, namespace="labels")]
                metric(class_probs, labels)
            output_dict['labels'] = labels

        if section_label is not None:  # this is the first scaffold task
            logits = self.classifier_feedforward_2(encoded_citation_text)
            class_probs = F.softmax(logits, dim=1)
            output_dict = {"logits": logits}
            loss = self.loss(logits, section_label)
            output_dict["loss"] = loss
            for i in range(self.num_classes_sections):
                metric = self.label_f1_metrics_sections[
                    self.vocab.get_token_from_index(index=i, namespace="section_labels")]
                metric(class_probs, section_label)

        if is_citation is not None:  # second scaffold task
            logits = self.classifier_feedforward_3(encoded_citation_text)
            class_probs = F.softmax(logits, dim=1)
            output_dict = {"logits": logits}
            loss = self.loss(logits, is_citation)
            output_dict["loss"] = loss
            for i in range(self.num_classes_cite_worthiness):
                metric = self.label_f1_metrics_cite_worthiness[
                    self.vocab.get_token_from_index(index=i, namespace="cite_worthiness_labels")]
                metric(class_probs, is_citation)

        if self.predict_mode:
            logits = self.classifier_feedforward(encoded_citation_text)
            # class_probs = F.softmax(logits, dim=1)
            output_dict = {"logits": logits}

        output_dict['citing_paper_id'] = citing_paper_id
        output_dict['cited_paper_id'] = cited_paper_id
        output_dict['citation_excerpt_index'] = citation_excerpt_index
        output_dict['citation_id'] = citation_id
        output_dict['attn_dist'] = attn_dist  # also return attention distribution for analysis
        output_dict['citation_text'] = citation_text['tokens']
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        class_probabilities = F.softmax(output_dict['logits'], dim=-1)
        predictions = class_probabilities.cpu().data.numpy()
        argmax_indices = np.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="labels")
                  for x in argmax_indices]
        output_dict['probabilities'] = class_probabilities
        output_dict['positive_labels'] = labels
        output_dict['prediction'] = labels
        citation_text = []
        for batch_text in output_dict['citation_text']:
            citation_text.append([self.vocab.get_token_from_index(token_id.item()) for token_id in batch_text])
        output_dict['citation_text'] = citation_text
        output_dict['all_labels'] = [self.vocab.get_index_to_token_vocabulary(namespace="labels")
                                     for _ in range(output_dict['logits'].shape[0])]
        return output_dict

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'CustomScaffoldBilstmAttentionClassifier':
        with_elmo = params.pop_bool("with_elmo", False)
        if with_elmo:
            embedder_params = params.pop("elmo_text_field_embedder")
        else:
            embedder_params = params.pop("text_field_embedder")
        with_bert = params.pop_bool("with_bert", False)
        if with_bert:
            bert_name = params.pop("bert_name")
            bert_model = AutoModel.from_pretrained(bert_name)
        else:
            bert_model = None

        text_field_embedder = TextFieldEmbedder.from_params(embedder_params, vocab=vocab)
        # citation_text_encoder = Seq2VecEncoder.from_params(params.pop("citation_text_encoder"))
        use_cnn = params.pop_bool("use_cnn", False)
        if use_cnn:
            citation_text_encoder = CNN.from_params(params.pop("citation_text_encoder"))
        else:
            citation_text_encoder = Seq2SeqEncoder.from_params(params.pop("citation_text_encoder"))
        classifier_feedforward = FeedForward.from_params(params.pop("classifier_feedforward"))
        classifier_feedforward_2 = FeedForward.from_params(params.pop("classifier_feedforward_2"))
        classifier_feedforward_3 = FeedForward.from_params(params.pop("classifier_feedforward_3"))

        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))

        use_lexicon = params.pop_bool("use_lexicon_features", False)
        use_sparse_lexicon_features = params.pop_bool("use_sparse_lexicon_features", False)
        use_pattern_features = params.pop_bool("use_pattern_features", False)
        weighted_loss = params.pop_bool("weighted_loss", False)
        focal_loss = params.pop_bool("focal_loss", False)
        tokenizer_len = int(params.pop("tokenizer_len"))
        use_mask = params.pop_bool("use_mask", False)
        use_cnn = params.pop_bool("use_cnn", False)
        data_format = params.pop('data_format')

        report_auxiliary_metrics = params.pop_bool("report_auxiliary_metrics", False)

        predict_mode = params.pop_bool("predict_mode", False)
        print(f"pred mode: {predict_mode}")

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   citation_text_encoder=citation_text_encoder,
                   classifier_feedforward=classifier_feedforward,
                   classifier_feedforward_2=classifier_feedforward_2,
                   classifier_feedforward_3=classifier_feedforward_3,
                   initializer=initializer,
                   regularizer=regularizer,
                   report_auxiliary_metrics=report_auxiliary_metrics,
                   predict_mode=predict_mode,
                   weighted_loss=weighted_loss,
                   focal_loss=focal_loss,
                   tokenizer_len=tokenizer_len,
                   bert_model=bert_model,
                   use_mask=use_mask,
                   use_cnn=use_cnn)


class CNN(nn.Module):

    def __init__(self,
                 embed_dim=768,
                 filter_sizes=None,
                 num_filters=None,
                 paddings=None,
                 padding_mode=None,
                 strides=None
                 ):
        super(CNN, self).__init__()
        # Embedding layer
        if num_filters is None:
            num_filters = [10, 10, 10]
        if filter_sizes is None:
            filter_sizes = [3, 4, 5]

        # Conv Network
        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim,
                      out_channels=num_filters[i],
                      padding=paddings[i],
                      padding_mode=padding_mode,
                      stride=strides[i],
                      kernel_size=filter_sizes[i])
            for i in range(len(filter_sizes))
        ])

    def forward(self, citation_text_embedding, citation_text_mask):
        # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
        x_conv_list = [F.relu(conv1d(citation_text_embedding)) for conv1d in self.conv1d_list]

        # Max pooling. Output shape: (b, num_filters[i], 1)
        x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
                       for x_conv in x_conv_list]

        # Concatenate x_pool_list to feed the fully connected layer.
        # Output shape: (b, sum(num_filters))
        x_cat = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list],
                         dim=1)
        return x_cat

    @classmethod
    def from_params(cls, params: Params) -> 'CNN':
        embed_dim = params.pop("embed_dim")
        filter_sizes = params.pop("filter_sizes")
        num_filters = params.pop("num_filters")
        paddings = params.pop("paddings")
        padding_mode = params.pop("padding_mode")
        strides = params.pop("strides")

        return cls(embed_dim,
                   filter_sizes,
                   num_filters,
                   paddings,
                   padding_mode,
                   strides)

    def get_output_dim(self) -> int:
        return 0
