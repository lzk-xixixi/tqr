import argparse
import json
import operator
import random
import numpy as np
import pandas as pd
from datetime import datetime
import pytz
from bs4 import BeautifulSoup
from lxml import etree
from random import shuffle
import time
import pickle
random.seed(1)
np.random.seed(1)

# sys.path.append("./")


parser = argparse.ArgumentParser(description='Bug Triaging Model')

parser.add_argument('--ratio_train', type=float, default=0.7, help='validation rate [default: 0.7]')
parser.add_argument('--ratio_val', type=float, default=0.1, help='validation rate [default: 0.1]')
parser.add_argument('--data_type', type=str, default='seq',
                    help='choose training type to run [options: seq, rand]')
parser.add_argument('--dataset', type=str, default='stackex_print',
                    help='choose dataset to run [options: stackex_ai, stackex_bioinformatics, stackex_3dprinting, '
                         'stackex_ebooks, stackex_history, stackex_philosophy]')
parser.add_argument('--data_path', type=str, default='../data/', help='the data directory')
parser.add_argument('--filter_num', type=int, default=1, help='the data directory')

args = parser.parse_args()



with open("../data/%s/q_titles_idx.pkl"%dataset,"rb") as f:
    q_titles_idx = pickle.load(f)

with open("../data/%s/q_titles_mask.pkl"%dataset,"rb") as f:
    q_titles_mask = pickle.load(f)
    
bert_embedder = PretrainedBertEmbedder(
                pretrained_model="bert-base-uncased",
            )
word_embeddings = BasicTextFieldEmbedder({"tokens": bert_embedder},
                                              allow_unmatched_keys=True)
encoder = BertSentencePooler(vocab, word_embeddings.get_output_dim())
result = encoder(q_titles_idx)