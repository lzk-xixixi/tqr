import math
import sys
from datetime import timedelta

import numpy as np
import seaborn
import torch
import torch.nn as nn
from torch.autograd import Variable

import br_utils

seaborn.set_context(context="talk")
sys.path.insert(0, "../")
from pandas import Timestamp


def gen_time_encoding(time_encoder, dates):
    time_embed = [time_encoder.get_time_encoding(i) for i in dates]
    time_embed = torch.stack(time_embed, dim=0)  # (n, d)
    return br_utils.to_variable(time_embed, requires_grad=False)


class BTE(torch.nn.Module):
    """
    bahurla time embedding
    """

    def __init__(self, expand_dim, factor=5):
        super(BTE, self).__init__()

        time_dim = expand_dim
        self.factor = factor
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim))).float())
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float())

    def forward(self, ts):
        batch_size = ts.size(0)
        seq_len = ts.size(1)
        ts = ts.view(batch_size, seq_len, 1)
        map_ts = ts * self.basis_freq.view(1, 1, -1)
        map_ts += self.phase.view(1, 1, -1)
        harmonic = torch.cos(map_ts)
        return harmonic


class TimeEncode(torch.nn.Module):
    def __init__(self, expand_dim, factor=5):
        super(TimeEncode, self).__init__()
        time_dim = expand_dim
        self.factor = factor
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim))).float())
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float())

    def forward(self, ts):
        batch_size = ts.size(0)
        seq_len = ts.size(1)
        ts = ts.view(batch_size, seq_len, 1)
        map_ts = ts * self.basis_freq.view(1, 1, -1)
        map_ts += self.phase.view(1, 1, -1)
        harmonic = torch.cos(map_ts)
        return harmonic 


class ContinueTimeEncoder:
    def __init__(self, d_model, factor, seg_num, upper, target):
        self.upper = upper
        self.seg_num = seg_num
        self.target = target
        self.timestamp_emb = BTE(d_model, factor)
        self.timeinterval_emb = BTE(d_model, factor)
        self.multi_date = self.make_time_seg(target, self.seg_num, self.upper)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.relu = nn.functional.relu
        conv_layer = 3
        kernel_sizes = [i * 2 + 1 for i in range(conv_layer)]
        pads = [int(i / 2) for i in kernel_sizes]
        num_channels = [d_model] * conv_layer
        self.convs = []
        self.convs1 = []
        for c, k, p in zip(num_channels, kernel_sizes, pads):
            self.convs.append(nn.Conv1d(in_channels=d_model,
                                        out_channels=c,
                                        kernel_size=k,
                                        padding=p
                                        ).to(self.device))
            self.convs1.append(nn.Conv1d(in_channels=d_model,
                                         out_channels=c,
                                         kernel_size=k,
                                         padding=p
                                         ).to(self.device))
        self.conv_last = nn.Conv1d(in_channels=d_model * conv_layer, out_channels=d_model, kernel_size=1)
        self.conv_last1 = nn.Conv1d(in_channels=d_model * conv_layer, out_channels=d_model, kernel_size=1)

    def time_seg(self, target, seg_num, upper):
        segment_list = []
        for j in range(seg_num - 1):
            segment_list.append(target ** (1 / (seg_num - 1)) ** j)
        segment_list.append(upper + 1)
        return segment_list

    def encode_multi_time(self, batch_time):
        interval = torch.tensor(self.multi_date).unsqueeze(0).expand(len(batch_time), -1)
        interval_emb = self.timeinterval_emb(interval)
        batch_time = torch.tensor([i.timestamp() / 60 for i in batch_time]).unsqueeze(1).expand(-1, interval.shape[1])
        multi_timstamp = self.timestamp_emb(batch_time + interval)
        multi_timstamp_dy = multi_timstamp.transpose(2, 1)
        multi_timstamp_dy = torch.cat([self.relu(conv(multi_timstamp_dy)).squeeze(-1) for conv in self.convs],dim=1)
        multi_timstamp_dy = self.conv_last(multi_timstamp_dy)
        multi_timstamp_dy = multi_timstamp_dy.transpose(2, 1)
        interval_emb_dy = interval_emb.transpose(2, 1)
        interval_emb_dy = torch.cat([self.relu(conv(interval_emb_dy)).squeeze(-1) for conv in self.convs1], dim=1)
        interval_emb_dy = self.conv_last1(interval_emb_dy)
        interval_emb_dy = interval_emb_dy.transpose(2, 1)
        return interval_emb, multi_timstamp,interval_emb_dy, multi_timstamp_dy


class TimeEncoder:
    def __init__(self, d_model, dropout, span, date_range):
        super(TimeEncoder, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.decay_rate = 1
        self.span = span
        self.from_date, self.to_date = date_range
        self.date_margin = 10
        max_time_span = self.temporal_index(self.to_date) + 1 + self.date_margin
        self.time_encode = torch.zeros(max_time_span, d_model)
        position = torch.arange(0., max_time_span).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * d_model)
        self.time_encode[:, 0::2] = torch.sin(position * div_term)
        self.time_encode[:, 1::2] = torch.cos(position * div_term)

    def temporal_index(self, date):
        month_span = (date.year - self.from_date.year) * 12 + date.month - self.from_date.month
        idx = int(month_span / self.span) + self.date_margin
        return idx

    def get_all_encoding(self):
        return self.time_encode

    def get_post_encodings(self, date):
        temp_idx = self.temporal_index(date)
        post_encodings = self.time_encode[temp_idx:, :]
        from br_utils import to_variable
        return to_variable(post_encodings, requires_grad=False)

    def get_time_encoding(self, date, num_shift=0):
        time_encode = self.time_encode[self.temporal_index(date), :]
        for i in range(1, num_shift + 1):
            time_encode_i_pre = self.time_encode[self.temporal_index(date) - i,:]
            time_encode_i_post = self.time_encode[self.temporal_index(date) + i, :]
            time_encode += (time_encode_i_pre + time_encode_i_post) * (
                    self.decay_rate ** i)
        time_encode /= (num_shift * 2 + 1)
        from br_utils import to_variable
        return to_variable(time_encode, requires_grad=False)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)