import math
from typing import Iterable
import sklearn
from allennlp.data import Instance
from allennlp.data.iterators import DataIterator, BasicIterator, BucketIterator
from allennlp.models import Model
from tqdm import tqdm
from scipy.special import expit  # the sigmoid function
import torch
import numpy as np
from allennlp.nn import util as nn_util
from br_utils import init_model, split
import json

np.random.seed(1)


def tonp(tsr): return tsr.detach().cpu().numpy()


def show_results(preds, truth, truth_answerer):
    mrr = metric_mrr(preds, truth)
    p1, p3, p5, p10 = metric_precisionk(preds, truth)
    ndcg = metric_ndcg(preds, truth_answerer)

    print('mrr: %.3f' % mrr, 'p1: %.3f' % p1, 'p3: %.3f' % p3, 'ndcg: %.3f' % ndcg, sep='\t')
    print('mrr: %.3f' % mrr, 'p1: %.3f' % p1, 'p3: %.3f' % p3, 'p5: %.3f' % p5, 'p10: %.3f' % p10, 'ndcg: %.3f' % ndcg, sep='\t')

def eval_model(model, vocab, test_ds, batch_size, device,test_num, question_candi=None, verbose=True):
    seq_iterator = BasicIterator(batch_size=batch_size)
    seq_iterator.index_with(vocab)
    predictor = Predictor(model, seq_iterator, test_num, cuda_device=device)
    metrics = model.get_metrics()
    test_preds, coherence = predictor.predic(test_ds, question_candi=question_candi)
    truth = [i["accept_usr"].metadata for i in test_ds]
    truth_answerer = [[ans[0] for ans in i["answerers"].metadata] for i in test_ds]


def eval_temp_model(model, vocab, test_ds, batch_size, device):
    seq_iterator = BasicIterator(batch_size=batch_size)
    seq_iterator.index_with(vocab)
    predictor = Predictor(model, seq_iterator, cuda_device=device)
    metrics = model.get_metrics()
    test_preds, coherences = predictor.predict(test_ds)
    answerer_indices = [i[0] for i in test_ds[0]["answerers"].metadata]
    temp_coherence = coherences[:, answerer_indices].transpose()
    for rec in temp_coherence:
        rec = [str(i) for i in rec.tolist()]
        print("\t".join(rec))


def metric_mrr(pred, truth):
    total_mrr, n = 0, len(truth)
    for i, label in enumerate(truth):
        cur_pred = pred[i, :]
        for rank, usr_idx in enumerate(cur_pred, start=1):
            if usr_idx in label:
                total_mrr += float(1 / rank)
                break
    return total_mrr / n

def metric_precisionk(pred, truth):
    top_n = [1, 3, 5, 10]
    topn_result = [0] * len(top_n)
    for i, labels in enumerate(truth):

        cur_pred = pred[i, :]
        for j, n in enumerate(top_n):
            hit = 1 if sum([1 for s in cur_pred[:n] if s in labels]) > 0 else 0
            topn_result[j] += hit

    return [(s / len(truth)) for s in topn_result]

def metric_ndcg(pred, truth_answerers):
    jump_flag = False
    total_ndcg, n = 0, len(truth_answerers)
    for i, truth_ans in enumerate(truth_answerers):
        truth_ans = [truth_ans[0]]
        dcg = 0
        cur_pred = pred[i, :]
        idcg = sum([1 / math.log2(j + 1) for j in range(1, len(truth_ans) + 1)])
        for rank, usr_idx in enumerate(cur_pred, start=1):
            if usr_idx in truth_ans:
                dcg += 1 / math.log2(rank + 1)
        if dcg > idcg:
            print("NDCG Error:")
            print(cur_pred, truth_ans)
            jump_flag = True
        if jump_flag:
            jump_flag = False
            continue
        total_ndcg += dcg / idcg
    return total_ndcg / n


def eval_ranking(coherence_rank, truth):
    recall_k = 0
    top_n = [1, 3, 5, 10]
    topn_result = [0] * len(top_n)
    for i, labels in enumerate(truth):
        rank = coherence_rank[i, :]
        k = len(labels)
        rk_hit = sum([1 for j in rank[:k] if j in labels]) / k
        recall_k += rk_hit
        for j, n in enumerate(top_n):
            hit = 1 if sum([1 for s in rank[:n] if s in labels]) > 0 else 0
            topn_result[j] += hit
    recall_k = recall_k / len(truth)
    result = [recall_k] + [(s / len(truth)) for s in topn_result]
    return result

class Predictor:
    def __init__(self, model: Model, iterator: DataIterator,
                 test_num,cuda_device: int = -1) -> None:
        self.model = model
        self.test_num = test_num
        self.iterator = iterator
        self.cuda_device = cuda_device

    def _extract_data(self, batch) -> np.ndarray:
        out_dict = self.model(**batch)
        return tonp(out_dict["coherence"])

    def predict(self, ds: Iterable[Instance], question_candi=None) -> np.ndarray:
        pred_generator = self.iterator(ds, num_epochs=1, shuffle=False)
        self.model.eval()
        pred_generator_tqdm = tqdm(pred_generator,
                                   total=self.iterator.get_num_batches(ds))
        preds, coherences = [], []

        with torch.no_grad():
            for batch in pred_generator_tqdm:
                batch = nn_util.move_to_device(batch, self.cuda_device)
                coherence = self._extract_data(batch)
                coherence_candi = np.zeros((len(batch["id"]), self.test_num))
                rank_candi = np.zeros((len(batch["id"]), self.test_num))
                candi_idx = np.array([question_candi[int(i)] for i in batch["id"]])
                for j in range(len(candi_idx)):
                    coherence_candi[j] = coherence[j][candi_idx[j]]
                preds.append(rank_candi)
                coherences.append(coherence_candi)
        mrr = self.model.get_metrics(reset=True)
        print("model metric:", mrr)
        return np.concatenate(preds, axis=0), np.concatenate(coherences, axis=0)