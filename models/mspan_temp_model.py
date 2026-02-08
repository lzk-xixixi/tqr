import copy
from typing import *

import torch
import torch.nn as nn
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedBertEmbedder
from allennlp.nn.util import get_text_field_mask
from overrides import overrides

import br_utils
from models.bert_sent_pooler import BertSentencePooler
# implementation of transformer encoder
from models.loss_modules import CoherenceLoss, TripletLoss, TemporalLoss, MarginRankLoss
from models.model_base import ModelBase
from models.multihead_ctx_model import TempCtxAttention
from models.metric_rank_recall import RankRecall
from models.shift_temp_model import ShiftTempAttention
from models.temp_ctx_model import MHTempCtxAttention
from models.time_encoder import TimeEncoder, gen_time_encoding
import numpy as np
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder, Embedding
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
# seaborn.set_context(context="talk")

'''
Starting my own MultiSpanTempModel
'''
class MultiSpanTempModel(ModelBase):
    def __init__(self, num_authors: int, out_sz: int,
                 vocab: Vocabulary, date_span: Any,
                 num_shift: int, spans: List, encoder: Any,
                 max_vocab_size: int, ignore_time: bool, ns_mode: bool=False, num_sk: int=20):
        super().__init__(vocab)

        self.date_span = date_span

        self.num_authors = num_authors

        self.num_sk, self.sk_dim = num_sk, 768
        self.ignore_time = ignore_time
        self.ns_mode = ns_mode
        if self.ns_mode:
            self.author_embeddings = nn.Parameter(torch.randn(num_authors, self.sk_dim),
                                                  requires_grad=True)  # (m, d)
        else:
            self.author_embeddings = nn.Parameter(torch.randn(num_authors, self.num_sk, self.sk_dim),
                                                  requires_grad=True)  # (m, k, d)


        self.encode_type = encoder
        if self.encode_type == "bert":
            bert_embedder = PretrainedBertEmbedder(
                pretrained_model="bert-base-uncased",
                top_layer_only=True,
            )
            self.word_embeddings = BasicTextFieldEmbedder({"tokens": bert_embedder},
                                                          allow_unmatched_keys=True)
            self.encoder = BertSentencePooler(vocab, self.word_embeddings.get_output_dim())
        else:
            token_embedding = Embedding(num_embeddings=max_vocab_size + 2,
                                        embedding_dim=300, padding_index=0)
            self.word_embeddings: TextFieldEmbedder = BasicTextFieldEmbedder({"tokens": token_embedding})

            self.encoder: Seq2VecEncoder = PytorchSeq2VecWrapper(nn.LSTM(self.word_embeddings.get_output_dim(),
                                                                         hidden_size=int(self.sk_dim / 2), bidirectional=True,
                                                                         batch_first=True))

        self.ctx_attention = TempCtxAttention(h=8, d_model=self.sk_dim)
        self.ctx_layer_norm = nn.LayerNorm(self.sk_dim)
        self.spans = spans
        self.span_temp_atts = nn.ModuleList()
        for span in self.spans:
            self.span_temp_atts.append(ShiftTempAttention(self.num_authors, self.sk_dim, date_span, num_shift, span, self.ignore_time))
        self.span_projection = nn.Linear(len(spans), 1)
        self.num_shift = num_shift
        self.time_encoder = TimeEncoder(self.sk_dim, dropout=0.1, span=spans[0], date_range=date_span)
        self.temp_loss = TemporalLoss(self.encoder.get_output_dim(), out_sz)
        self.rank_loss = MarginRankLoss(self.encoder.get_output_dim(), out_sz)
        self.weight_temp = 0.3
        self.visual_id = 0

    def forward(self, tokens: Dict[str, torch.Tensor],
                id: Any, answerers: Any, date: Any, accept_usr: Any) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        embeddings = self.word_embeddings(tokens)
        token_hidden2 = self.encoder(embeddings, mask)
        if self.encode_type == "bert":
            token_hidden = self.encoder(embeddings, mask).transpose(-1, -2)
            token_embed = torch.mean(token_hidden, 1).squeeze(1)
        else:
            token_embed = self.encoder(embeddings, mask)
        time_embed = gen_time_encoding(self.time_encoder, date)
        token_temp_embed = token_embed if self.ignore_time else token_embed + time_embed
        if self.ns_mode:
            author_ctx_embed = self.author_embeddings.unsqueeze(0).expand(token_embed.size(0), -1, -1)
        else:
            author_ctx_embed = self.ctx_attention(token_temp_embed, self.author_embeddings, self.author_embeddings)
            author_ctx_embed = self.ctx_layer_norm(author_ctx_embed)

        span_temp_ctx_embeds, history_embeds = [], []
        for i in range(len(self.spans)):
            temp_ctx_embed, history_embed = self.span_temp_atts[i](token_embed, author_ctx_embed, date)
            span_temp_ctx_embeds.append(temp_ctx_embed)
            history_embeds.append(history_embed)
        temp_ctx_embed_sp = torch.stack(span_temp_ctx_embeds, dim=-1)
        temp_ctx_embed = torch.squeeze(self.span_projection(temp_ctx_embed_sp), dim=-1)
        triplet_loss, coherence = self.rank_loss(token_embed, temp_ctx_embed, answerers, accept_usr)
        truth = [[j[0] for j in i] for i in answerers]
        if self.num_shift > 2:
            temp_loss = sum([self.temp_loss(token_embed, history_embed, truth) for history_embed in history_embeds])
        else:
            temp_loss = 0
        loss = triplet_loss + temp_loss * self.weight_temp
        output = {"loss": loss, "coherence": coherence}
        predict = np.argsort(-coherence.detach().cpu().numpy(), axis=1)
        self.mrr(predict, accept_usr)
        return output