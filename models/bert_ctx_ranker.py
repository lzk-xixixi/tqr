import random
from typing import *

import numpy as np
import torch
import torch.nn as nn
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import Seq2VecEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedBertEmbedder
from allennlp.nn.util import get_text_field_mask
from overrides import overrides
from torch.autograd import Variable
from statistics import mean

from models.bert_sent_pooler import BertSentencePooler
from models.loss_modules import CoherenceLoss, TripletLoss, MarginRankLoss
from models.model_base import ModelBase


class BertCtxRanker(ModelBase):
    def __init__(self, args, num_authors: int, out_sz: int,
                 vocab: Vocabulary):
        super().__init__(vocab)

        # init word embedding
        bert_embedder = PretrainedBertEmbedder(
            pretrained_model="bert-base-uncased",
            top_layer_only=True,
        )
        self.word_embeddings = BasicTextFieldEmbedder({"tokens": bert_embedder},
                                                      allow_unmatched_keys=True)

        self.encoder = BertSentencePooler(vocab, self.word_embeddings.get_output_dim())
        self.num_authors = num_authors
        self.num_sk, self.sk_dim = 20, 768
        self.author_embeddings = nn.Parameter(torch.randn(num_authors, self.sk_dim, self.num_sk), requires_grad=True)
        self.attention = nn.Parameter(torch.randn(self.word_embeddings.get_output_dim(), self.sk_dim), requires_grad=True)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()
        self.projection = nn.Linear(self.encoder.get_output_dim(), out_sz)
        self.triplet_loss = TripletLoss(self.encoder.get_output_dim(), out_sz)
        self.rank_loss = MarginRankLoss(self.encoder.get_output_dim(), out_sz)

    def build_author_ctx_embed(self, token_hidden, author_embeds):
        n, _, l = token_hidden.shape
        m = author_embeds.shape[0]

        F_sim = torch.einsum('ndl,de,mek->nmlk', [token_hidden, self.attention, author_embeds])
        F_tanh = self.tanh(F_sim.contiguous().view(n * m, l, self.num_sk))
        F_tanh = F_tanh.view(n, m, l, self.num_sk)
        g_u = torch.mean(F_tanh, 2)
        a_u = self.softmax(g_u)

        author_ctx_embed = torch.einsum('mdk,nmk->nmd', [author_embeds, a_u])

        return author_ctx_embed

    def forward(self, tokens: Dict[str, torch.Tensor],
                id: Any, answerers: Any, date: Any, accept_usr: Any, att_l=False) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        embeddings = self.word_embeddings(tokens)
        token_hidden = self.encoder(embeddings, mask)
        author_ctx_embed = self.build_author_ctx_embed(token_hidden, self.author_embeddings)
        token_embed = torch.mean(token_hidden, 2)
        loss, coherence = self.rank_loss(token_embed, author_ctx_embed, answerers, accept_usr)
        output = {"loss": loss, "coherence": coherence}
        
        predict = np.argsort(-coherence.detach().cpu().numpy(), axis=1)
        truth = [[j[0] for j in i] for i in answerers]


        self.mrr(predict, accept_usr)
        return output
