import argparse
import copy
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import torch.optim as optim
from GPUtil import GPUtil
from allennlp.data.iterators import BucketIterator, BasicIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.trainer import Trainer

from br_utils import init_model, complete_snapshot, detect_model_from_path
# from data_process.br_data_reader import load_br_data
from data_process.stackex_data_reader import load_stackex_data
from eval import eval_model, eval_temp_model
from pandas import Timestamp

logging.basicConfig(level=logging.ERROR)
parser = argparse.ArgumentParser(description='Bug Triaging Model')
parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate [default: 1e-4]')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay [default: 0]')
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs for train [default: 1000]')
parser.add_argument('--batch_size', type=int, default=64, help='batch size for training [default: 64]')
parser.add_argument('--seed', type=int, default=1, help='seed of random numbers [default: 1]')
parser.add_argument('--device', type=int, default=0, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('--toy_data', action='store_true', default=False,
                    help='use an extremely small dataset for testing purpose [default: False]')
parser.add_argument('--data_type', type=str, default='seq',
                    help='choose training type to run [options: seq, cross, rand, single]')
parser.add_argument('--max_seq_len', type=int, default=300,
                    help='maximum sequence length which is necessary to limit memory usage [default: 200]')
parser.add_argument('--test_num', type=int, default=100,)
parser.add_argument('--max_vocab_size', type=int, default=100000, help='maximum vocabulary size [default: 100000]')
parser.add_argument('--is_rank_task', type=bool, default=True,
                    help='whether use the ranking task, otherwise it is classification task [default: True]')
# datasets
parser.add_argument('--dataset', type=str, default='stackex_',
                    help='choose dataset to run [options: HIVE, COLLECTIONS, COMPRESS, stackex_ai, stackex_bioinformatics, '
                         'stackex_3dprinting, stackex_ebooks, stackex_history, stackex_philosophy]')
# model
parser.add_argument('--model', type=str, default="mspan_temp",
                    help='models: mspan_temp, shift_temp, tempx_ctx, tempq_ctx, temp_ctx, mh_ctx, bert_ctx, bert_noctx, '
                         'lstm_classifier, bert_classifier [default: bert_classifier]')
parser.add_argument('--bio', type=str, default="",
                    help='short bio to distinguish other models [default: ""]')
parser.add_argument('--num_sk', type=int, default=20, help='the number of user expertise areas [default: 20]')

parser.add_argument('--num_shift', type=int, default=3, help='the number of shifted time encoding [default: 10]')
parser.add_argument('--spans', type=str, default="1,2,3", help='the list string of spans [default: 1,2,3]')
parser.add_argument('--ignore_time', action='store_true', default=False,
                    help='For ablation study: whether to use time in the model')

# data
parser.add_argument('--data_path', type=str, default='./data/', help='the data directory')
parser.add_argument('--encoder', type=str, default='bert', help='the type of encoder, bert or lstm')
parser.add_argument('--snapshot_path', type=str, default="./snapshot/",
                    help='path of snapshot models [default: ./snapshot/]')

# testing
# parser.add_argument('--test', action='store_true', default=False, help='use test mode')
parser.add_argument('--test', default=False, help='use test mode')
parser.add_argument('--temp_test', action='store_true', default=False, help='use temporal test mode')
parser.add_argument('--snapshot', type=str, default='best.th',
                    help='filename of model snapshot, if only directory is given, use the best.th [default: None]')

# bundle testing
parser.add_argument('--bt', action='store_true', default=False, help='use bundle test mode')
parser.add_argument('--bt_path', type=str, default=None, help='file folder of model snapshots [default: None]')
parser.add_argument('--bt_max', type=int, default=100, help='maximum models are tested in the bt mode [default: 50]')

args = parser.parse_args()

# update args and print
args.cuda = torch.cuda.is_available()

PROJECT = args.dataset
# DATA_ROOT = Path("../dataset/bugtriaging/") / PROJECT
DATA_ROOT = Path(args.data_path) / PROJECT
USER_ID_FILE = DATA_ROOT / "user_ids.txt"
torch.manual_seed(args.seed)

# initialize the snapshot path
now = datetime.now()  # current date and time
date_time = now.strftime("%Y%m%d_%H%M%S")
bio_str = "[" + args.bio + "]" if args.bio else ""
if not args.test and not args.temp_test:
    args.snapshot_path = args.snapshot_path + args.model + bio_str + "_" + PROJECT + "_" + args.data_type + "_" + date_time + "/"
if not args.toy_data and not args.test and not os.path.isdir(args.snapshot_path):
    os.makedirs(args.snapshot_path)

if len(args.spans) > 0:
    args.spans = [int(i) for i in args.spans.split(",")]
else:
    args.spans = []

print("========== Parameter Settings ==========")
for arg in vars(args):
    print(arg, "=", getattr(args, arg), sep="")
print("========== ========== ==========")

# load data
print("Loading data:", DATA_ROOT, " (", args.data_type, ")", sep="")
reader, train_ds, val_ds, test_ds = load_stackex_data(DATA_ROOT, USER_ID_FILE, args.encoder, args.data_type,
                                                      args.max_seq_len, args.toy_data)

num_authors = reader.get_author_num()
print("User Number: ", str(num_authors))

try:
    if args.device < 0:
        avail_gpus = GPUtil.getFirstAvailable(maxMemory=0.1, maxLoad=0.05)
        if len(avail_gpus) == 0:
            print("No GPU available!!!")
            sys.exit()
        args.device = avail_gpus[0]
        print("### Use GPU", str(args.device), "###")
    torch.cuda.set_device(args.device)
except Exception as e:
    pass
vocab = Vocabulary.from_instances(train_ds, max_vocab_size=args.max_vocab_size)
iterator = BasicIterator(batch_size=args.batch_size)
iterator.index_with(vocab)

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

date_span = reader.get_time_span()
print("Date span: ", date_span[0], "--", date_span[1])
if not args.test and not args.bt and not args.temp_test:
    print("init Model ...")
    model = init_model(args, args.model, num_authors, vocab, args.encoder, args.max_vocab_size, date_span,
                       args.ignore_time, args.num_sk,args.dataset)

    # move model to gpu
    if args.cuda:
        model.cuda()

    # Train
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    serialization_dir = args.snapshot_path if not args.toy_data else None
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        iterator=iterator,
        train_dataset=train_ds,
        serialization_dir=serialization_dir,
        validation_dataset=val_ds,
        validation_metric="+mrr",
        cuda_device=args.device if args.cuda else -1,
        num_epochs=args.epochs,
        num_serialized_models_to_keep=0,
        # grad_norm=0.5,
        # grad_clipping=1
    )
    print("Training Model ...")
    metrics = trainer.train()