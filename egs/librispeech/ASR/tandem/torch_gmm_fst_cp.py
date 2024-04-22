#!/usr/bin/env python
# coding: utf-8

import sys

#sys.path.append('/home/jovyan/icefall')

import k2
import torch
# from icefall.lexicon import Lexicon
import numpy as np
from lhotse.supervision import SupervisionSegment, SupervisionSet
from tqdm import tqdm
from torch.nn import Module
from torch import nn
from torch.optim import Adam

from lhotse.cut import CutSet
from lhotse import load_manifest, fix_manifests
from torch.utils.data import DataLoader
from lhotse.dataset import K2SpeechRecognitionDataset, SimpleCutSampler
from lhotse.dataset.input_strategies import AudioSamples, PrecomputedFeatures
from k2.autograd import intersect_dense
from icefall.utils import encode_supervisions

from icefall.bpe_graph_compiler import BpeCtcTrainingGraphCompiler


def compute_gmm_pdf_per_sample(seq, mu, log_cov, log_pi):
    B, T, F = seq.shape

    D = mu.shape[2]  # fea dimension
    L = mu.shape[1]  # num Gaussians
    K = mu.shape[0]  # num BPE

    # KxL constant + log_det terms for each BPE and per-bpe gaussian
    C = -torch.ones((K, L), device=mu.device) * D / 2 * torch.log(
        torch.tensor([2 * np.pi], device=mu.device)) - 0.5 * torch.sum(log_cov, -1) # why 0.5
    CC = log_pi - torch.logsumexp(log_pi, dim=-1).unsqueeze(-1)

    seq_reshaped = seq.contiguous().view(B * T, F)

    per_elem_bpe_likes = []
    for i in range(seq_reshaped.shape[0]):
        seq_centered = seq_reshaped[i].unsqueeze(0).unsqueeze(0) - mu

        # KxL
        ULL = 0.5 * torch.sum(seq_centered * seq_centered / torch.exp(log_cov), -1)
        LL = torch.logsumexp(C - ULL + CC, -1)
        per_elem_bpe_likes.append(LL)
    return torch.stack(per_elem_bpe_likes).contiguous().view(B, T, -1)


class GMM(Module):
    def __init__(self, num_gauss, num_bpe, init_mean, init_std=1, fea_dim=1024):
        super().__init__()

        self.mu = nn.Parameter(
            torch.rand((num_bpe, num_gauss, fea_dim)) * init_std + init_mean.unsqueeze(0).unsqueeze(0),
            requires_grad=True)
        self.log_cov = nn.Parameter(torch.log(torch.ones((num_bpe, num_gauss, fea_dim))), requires_grad=True)
        self.log_pi = nn.Parameter(torch.log(torch.ones((num_bpe, num_gauss)) / num_gauss), requires_grad=True)

    def forward(self, X):
        return compute_gmm_pdf_per_sample(X, self.mu, self.log_cov, self.log_pi)


# model = Gauss(499)
model = GMM(10, 499,torch.zeros(499, 10, 1024))
# device = torch.device('cuda')
device = torch.device('cpu')
model.to(device)


opt = Adam(model.parameters(), lr=1e-3)



graph = BpeHMMTrainingGraphCompiler('/home/jovyan/icefall/egs/librispeech/ASR/data/lang_bpe_500', topo_type='hmm',
                                    device=device)
# graph = BpeCtcTrainingGraphCompiler('/home/jovyan/icefall/egs/librispeech/ASR/data/lang_bpe_500', device=device)

cs = CutSet.from_jsonl('/home/jovyan/wavlm_hmm/eval2000/data/eval2000/lhotse_cset.jsonl')

dataset = K2SpeechRecognitionDataset(cs, input_strategy=PrecomputedFeatures())
sampler = SimpleCutSampler(cs, shuffle=False, max_cuts=1)
loader = DataLoader(dataset, sampler=sampler, batch_size=None, num_workers=4, prefetch_factor=4)

for ep in range(30):
    for batch in tqdm(loader):
        with torch.amp.autocast(device_type='cuda', enabled=False):
            for i in range(10000):
                opt.zero_grad()

                net_out = model(batch['inputs'].to(device))
                supervision_segments, texts = encode_supervisions(batch['supervisions'], 1)
                ids = graph.texts_to_ids(texts)
                hmm_graph = graph.compile(ids)

                dense_fsa_vec = k2.DenseFsaVec(
                    net_out,
                    supervision_segments.cpu()
                )

                lattice = intersect_dense(hmm_graph, dense_fsa_vec, output_beam=8)

                nll = -(lattice.get_tot_scores(log_semiring=True, use_double_scores=True) / batch['supervisions'][
                    'num_frames'].to(device)).sum()

                nll.backward()

                opt.step()

                print(nll.item())
                # break
        break
    break