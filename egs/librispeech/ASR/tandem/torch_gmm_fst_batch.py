#!/usr/bin/env python
# coding: utf-8

import sys

# sys.path.append('/home/jovyan/icefall')

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
import sentencepiece as spm
# from icefall.bpe_graph_compiler import BpeCtcTrainingGraphCompiler
from rui_bpe_graph_compiler import BpeCtcTrainingGraphCompiler
import logging
from collections import defaultdict
from torch_scatter import scatter_add
from GMM import GMM_dominik as GMM
from GMM import Gauss

logging.basicConfig(level=logging.INFO)

# Example log messages
# logging.debug('This is a debug message')
# logging.info('This is an info message')
# logging.warning('This is a warning message')
# logging.error('This is an error message')
# logging.critical('This is a critical message')

# fixed random seed
torch.manual_seed(0)
np.random.seed(0)

def main():
    # define the gauss layer
    num_bpe = 500
    fea_dim = 1024
    init_mean = torch.zeros(num_bpe, fea_dim)
    gmm_layer = Gauss(num_bpe, init_mean, init_std=1,
                      fea_dim=fea_dim)  # single gaussian, with # bpe tokens = 500 including the blank
    # gmm_layer = GMM(10, num_bpe,torch.zeros(num_bpe, 10, 1024))

    # define the device
    device = torch.device('cpu')
    # move the gauss layer to the device
    gmm_layer.to(device)

    graph = BpeCtcTrainingGraphCompiler('/export/fs05/mliu121/icefall/egs/librispeech/ASR/data/lang_bpe_500',
                                        topo_type='hmm',
                                        device=device)
    logging.info('number of tokens: %s', graph.max_token_id + 1)
    # graph = BpeCtcTrainingGraphCompiler('/home/jovyan/icefall/egs/librispeech/ASR/data/lang_bpe_500', device=device)

    # load the cut set
    cs = CutSet.from_jsonl('/export/fs05/dklemen1/kaldi/egs/swbd_wavlm/s5c/data/eval2000/lhotse_cset.jsonl')

    dataset = K2SpeechRecognitionDataset(cs, input_strategy=PrecomputedFeatures())
    sampler = SimpleCutSampler(cs, shuffle=False, max_cuts=10)  # batch size is 1
    loader = DataLoader(dataset, sampler=sampler, batch_size=None, num_workers=4, prefetch_factor=4)

    batch = next(iter(loader))

    # for ep in range(0):
    #     for batch in tqdm(loader):
    for i in range(10):
    #for batch in tqdm(loader):
        logging.info('the batch is %s', batch['inputs'].shape)
        log_likelihood = train_one_batch(batch, gmm_layer, graph, device, num_bpe, fea_dim)
        logging.info('log_likelihood for each batch: %s', log_likelihood)
def train_one_batch(batch, gmm_layer, graph, device, num_bpe=500, fea_dim=1024):
    '''

    Args:
        batch: a dict containing the inputs and supervisions
        gmm_layer: the GMM layer
        graph: the BpeCtcTrainingGraphCompiler
        device: the device to run the model
        num_bpe: the number of bpe tokens
        fea_dim: the feature dimension
    return:
      the likelihood of the batch
    '''
    net_out = gmm_layer(batch['inputs'].to(device))  # BxTxK
    logging.info('successfully forward the batch through the GMM layer')
    batch_size = net_out.shape[0]
    num_frames = net_out.shape[1]
    supervision_segments, texts = encode_supervisions(batch['supervisions'], 1)

    ids = graph.texts_to_ids(texts)
    hmm_graph = graph.compile(ids)

    dense_fsa_vec = k2.DenseFsaVec(
        net_out,  # shape BxTxK
        supervision_segments.cpu()  # shape BX3, where the 3 column represents the sequence_index, start_frame, duration
    )

    lattice = intersect_dense(hmm_graph, dense_fsa_vec, output_beam=8)

    # nll = -(lattice.get_tot_scores(log_semiring=True, use_double_scores=True) / batch['supervisions'][
    #     'num_frames'].to(device)).sum()
    tot_score = lattice._get_tot_scores(log_semiring=True, use_double_scores=True)

    # valid_arcs,there are total num_arcs=508 arcs, but only those whose values are not -1 are valid, coolect the valid arcs index
    arc_values = lattice.arcs.values()
    valid_arcs_mask = arc_values[:, 2] != -1
    num_valid_arcs = int(torch.sum(valid_arcs_mask))


    arc_post = lattice._get_arc_post(log_semiring=True, use_double_scores=True)  # 74
    # use the valid_arcs_mask to filter the arc_post
    arc_post = arc_post[valid_arcs_mask]
    num_arcs = lattice.num_arcs

    # by combining the lattice.arcs.row_ids(2) and lattice.arcs.row_ids(1), we can get to which batch that this arc belongs to
    state_index = lattice.arcs.row_ids(2).long()  # to which state that this arc belongs to
    batch_index = lattice.arcs.row_ids(1)[state_index].long()  # to which batch that this arc belongs to
    # use the valid_arcs_mask to filter the batch_index
    batch_index = batch_index[valid_arcs_mask]

    #  by combining the lattice.state_batches.shape.row_ids(2) and lattice.state_batches.shape.row_ids(1), we can get to time frame that this state belongs to
    # this is not working, because the two fsas are mixed together
    state_rowid2 = lattice.state_batches.shape.row_ids(2).long()
    state_time_frame = lattice.state_batches.shape.row_ids(1)[
        state_rowid2].long()  # to which time frame that this state belongs to
    flat_values = lattice.state_batches.values.long()  # the corresponding state index

    # initialize the time_index, with the shape same as state_time_frame
    state_time_index = torch.ones_like(state_time_frame)
    # reorgnize the state_time_frame based on the flat_value
    state_time_index[flat_values] = state_time_frame  # to which time frame that this state belongs to
    time_index = state_time_index[state_index]  # from which time frame that this arc belongs to
    # use the valid_arcs_mask to filter the time_index
    time_index = time_index[valid_arcs_mask]

    bpe_index = lattice.arcs.values()[:, 2].long()  # the bpe index of each arc
    # use the valid_arcs_mask to filter the bpe_index
    bpe_index = bpe_index[valid_arcs_mask]

    index = batch_index * num_frames * num_bpe + time_index * num_bpe + bpe_index  # the absolute index of the tensor time_index*num_bpe+bpe_index
    temp = torch.zeros((batch_size * num_frames * num_bpe)).double()  # T*K
    temp = scatter_add(torch.exp(arc_post), index, out=temp)
    temp = torch.log(temp.view(batch_size, num_frames, num_bpe))
    log_post_tensor_2 = temp

    # update the mu and cov based on the log_post_tensor
    post_tensor = torch.exp(log_post_tensor_2)

    # shape of batch['inputs']: BxTxF 1,42,1024
    # shape of post_tensor: BxTxK 1,42,500
    # shape of mu and cov: KxF 500,1024
    norm = torch.sum(post_tensor, dim=(0, 1)) + 1e-5  # K
    mu = torch.sum(post_tensor.unsqueeze(-1) * batch['inputs'].unsqueeze(2), dim=(0, 1)) / norm.unsqueeze(-1)

    frame_mu = batch['inputs'].unsqueeze(2) - mu  # BxTxKxF
    cov = torch.sum(post_tensor.unsqueeze(-1) * (frame_mu * frame_mu), dim=(0, 1)) / norm.unsqueeze(-1) + 1e-5
    # make sure that mu and cov are not inf or nan
    assert torch.isfinite(mu).all(), mu
    assert torch.isfinite(cov).all(), cov
    gmm_layer.mu = nn.Parameter(mu.float(), requires_grad=False)
    gmm_layer.cov = nn.Parameter(cov.float(), requires_grad=False)
    #return mu, cov
    return tot_score.mean()


if __name__ == '__main__':
    # run the main function
    main()
