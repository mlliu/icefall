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
import sentencepiece as spm
# from icefall.bpe_graph_compiler import BpeCtcTrainingGraphCompiler
from rui_bpe_graph_compiler import BpeCtcTrainingGraphCompiler
import logging
from collections import defaultdict
from torch_scatter import scatter_add


logging.basicConfig(level=logging.DEBUG)

# Example log messages
# logging.debug('This is a debug message')
# logging.info('This is an info message')
# logging.warning('This is a warning message')
# logging.error('This is an error message')
# logging.critical('This is a critical message')

# fixed random seed
torch.manual_seed(0)
np.random.seed(0)

class Gauss(nn.Module):
    def __init__(self, num_bpe, fea_dim=1024):
        super().__init__()

        self.mu = nn.Parameter(torch.rand((num_bpe, fea_dim)), requires_grad=True)
        self.cov = nn.Parameter(torch.ones((num_bpe, fea_dim)), requires_grad=True)

    def forward(self, X):
        """
        input: X: BxTxF, B is the batch size, T is the number of frames, F is the feature dim
        mu: KxF, K is the number of BPEs, F is the feature dim
        cov: KxF
        out: BxTxK - likelihoods
        """
        return compute_pdf_batch(X, self.mu, self.cov)


def compute_pdf(seq, mu, cov):
    """
    input: seq: TxF
    mu: KxF
    cov: KxF
    out: Tx1 - likelihoods
    """

    D = mu.shape[-1]
    K = mu.shape[0]
    # 1xK tensor
    C = -torch.ones((K,), device=mu.device) * D / 2 * torch.log(
        torch.tensor([2 * np.pi], device=mu.device)) - 0.5 * torch.sum(torch.log(cov), -1)

    # unnormalized log likelihood - TxKxF
    seq_centered = (seq.unsqueeze(1) - mu.unsqueeze(0))
    ULL = 0.5 * torch.sum(seq_centered * seq_centered / torch.unsqueeze(cov, 0), -1)
    return torch.unsqueeze(C, 0) - ULL


def compute_pdf_batch(seq, mu, cov):
    """
    input: seq: TxF
    mu: KxF
    cov: KxF
    out: Tx1 - likelihoods
    """

    D = mu.shape[-1]
    K = mu.shape[0]
    # 1xK tensor
    C = -torch.ones((K,), device=mu.device) * D / 2 * torch.log(
        torch.tensor([2 * np.pi], device=mu.device)) - 0.5 * torch.sum(torch.log(cov), -1)

    # unnormalized log likelihood - TxKxF
    seq_centered = (seq.unsqueeze(2) - mu.unsqueeze(0).unsqueeze(0))
    ULL = 0.5 * torch.sum(seq_centered * seq_centered / torch.unsqueeze(cov, 0).unsqueeze(0), -1)
    return torch.unsqueeze(C, 0).unsqueeze(0) - ULL


# def EM_update(lattice):
#     '''
#     perform EM update on the GMM parameters
#     Args:
#         lattice: the fsa vec built from the GMM output and the supervision segments using the HMM graph compiler
#
#     Returns:
#         updated mu, cov, log_pi
#     '''
#
#     # get the forward probs
#     forward_prob = lattice._get_forward_scores(log_semiring=True, use_double_scores=True)
#     backward_prob = lattice._get_backward_scores(log_semiring=True, use_double_scores=True)
#     tot_scores = lattice.get_tot_scores(log_semiring=True, use_double_scores=True)
#
#     # get the posteriors
#     posteriors = forward_prob + backward_prob - tot_scores
#
#     # get the arc posteriors
#     arc_posteriors = lattice.get_arc_post(log_semiring  = True, use_double_scores = True)


def main():
    # define the gauss layer
    num_bpe = 500
    gmm_layer = Gauss(num_bpe) # single gaussian, with # bpe tokens = 500 including the blank
    # define the device
    device = torch.device('cpu')
    # move the gauss layer to the device
    gmm_layer.to(device)

    graph = BpeCtcTrainingGraphCompiler('/export/fs05/mliu121/icefall/egs/librispeech/ASR/data/lang_bpe_500', topo_type='hmm',
                                    device=device)
    logging.info('number of tokens: %s', graph.max_token_id+1)
    # graph = BpeCtcTrainingGraphCompiler('/home/jovyan/icefall/egs/librispeech/ASR/data/lang_bpe_500', device=device)

    # load the cut set
    cs = CutSet.from_jsonl('/export/fs05/dklemen1/kaldi/egs/swbd_wavlm/s5c/data/eval2000/lhotse_cset.jsonl')

    dataset = K2SpeechRecognitionDataset(cs, input_strategy=PrecomputedFeatures())
    sampler = SimpleCutSampler(cs, shuffle=False, max_cuts=1) # batch size is 1
    loader = DataLoader(dataset, sampler=sampler, batch_size=None, num_workers=4, prefetch_factor=4)

    batch = next(iter(loader))
    # use the log to print the batch
    logging.debug('batch supervision: %s', batch['supervisions']['text'] )
    logging.debug('batch supervision num_frames: %s', batch['supervisions']['num_frames'])
    logging.debug('batch inputs: %s', batch['inputs'].shape)

    net_out = gmm_layer(batch['inputs'].to(device)) # BxTxK
    batch_size= net_out.shape[0]
    num_frames= net_out.shape[1]



    logging.debug('net_out: %s', net_out.shape)
    supervision_segments, texts = encode_supervisions(batch['supervisions'], 1)
    logging.debug('supervision_segments: %s', supervision_segments)
    logging.debug('texts: %s', texts)
    ids = graph.texts_to_ids(texts)
    hmm_graph = graph.compile(ids)
    # draw the hmm graph, but since hmm_graph is a fsavec
    hmm_graph_fsa= hmm_graph[0]
    hmm_graph_fsa.labels_sym = graph.word_table
    hmm_graph_fsa.aux_labels_sym = graph.word_table
    # hmm_graph_fsa.draw('graph/hmm_graph.pdf', title='HMM graph for _O h _YE A H')
    #graph.topo.labels_sym = graph.word_table
    # graph.topo.draw('graph/hmm_topo.pdf')
    # hmm_topo_toy=BpeCtcTrainingGraphCompiler.hmm_topo(3, [0, 1, 3], device=device)
    #hmm_topo_toy.labels_sym = graph.word_table
    sym_str = '''
      _O 1
      H 2
      _YE 3
    '''
    # hmm_topo_toy.labels_sym = k2.SymbolTable.from_str(sym_str)
    # hmm_topo_toy.aux_labels_sym = k2.SymbolTable.from_str(sym_str)
    #hmm_topo_toy.draw('graph/hmm_topo_toy.pdf',title='A toy HMM topo')

    logging.debug('hmm_topo shape: %s and length of the shape: %s', graph.topo.shape, len(graph.topo.shape))
    logging.debug('hmm_graph shape: %s and length of the shape: %s', hmm_graph.shape, len(hmm_graph.shape))




    dense_fsa_vec = k2.DenseFsaVec(
        net_out, # shape BxTxK
        supervision_segments.cpu() # shape BX3, where the 3 column represents the sequence_index, start_frame, duration
    )
    scores=dense_fsa_vec.scores
    logging.debug('dense_fsa_vec scores: %s', scores.shape) # (43, 500)# (43, 500), the last one is -1
    logging.debug('net_out: %s', net_out.shape) # (1,42,499), 499 is wrong, should be 500
    logging.debug('dense_fsa_vec: %s', dense_fsa_vec.dense_fsa_vec.shape)



    lattice = intersect_dense(hmm_graph, dense_fsa_vec, output_beam=100)
    # lattice is a fsavec, with len(lattice.shape)=2
    logging.debug('lattice shape: %s and length of the shape: %s', lattice.shape, len(lattice.shape))
    # lattice_fsa= lattice[0]
    # lattice_fsa.labels_sym = graph.word_table
    # lattice_fsa.aux_labels_sym = graph.word_table
    # lattice_fsa.draw('graph/lattice_beam100.pdf', title='Lattice with beam 100')

    # nll = -(lattice.get_tot_scores(log_semiring=True, use_double_scores=True) / batch['supervisions'][
    #     'num_frames'].to(device)).sum()
    tot_score = lattice._get_tot_scores(log_semiring=True, use_double_scores=True)
    forward_score = lattice._get_forward_scores(log_semiring=True, use_double_scores=True)
    backward_score = lattice._get_backward_scores(log_semiring=True, use_double_scores=True)
    arc_post = lattice._get_arc_post(log_semiring=True, use_double_scores=True) # 74
    num_arcs=lattice.num_arcs
    logging.debug('tot_score: %s', tot_score)
    logging.debug('forward_score: %s', forward_score.shape) # len=65 for each state
    # logging.debug('forward_score: %s', forward_score)
    logging.debug('backward_score: %s', backward_score.shape)# len=65 for each state
    # logging.debug('backward_score: %s', backward_score)
    logging.debug('arc_post: %s', arc_post)
    logging.debug('arc_post. shape: %s', arc_post.shape) # len=74 for each arc
    logging.debug('num_arcs: %s', num_arcs) # 74
    logging.debug('arcs.values(): %s', lattice.arcs.values()) # (source state, dest state, label, score_int) len=74
    # logging.debug('scores: %s', lattice.scores)
    #logging.debug('labels: %s', lattice.labels) # the labels of the arcs is
    # logging.debug('properties: %s', lattice.properties)  # the labels of the arcs is
    # logging.debug('row_splits: %s', lattice[0].shape)
    # logging.debug('row_splits(1): %s', lattice.arcs.row_splits(1))
    # logging.debug('row_splits(2): %s', lattice.arcs.row_splits(2))
    # logging.debug('row_ids(1): %s', lattice.arcs.row_ids(1))
    logging.debug('row_ids(2): %s', lattice.arcs.row_ids(2)) # from this we can get to which state that this arc belongs to
    logging.debug('row_ids(2).shape: %s', lattice.arcs.row_ids(2).shape)

    # logging.debug('state_batches: %s', lattice.state_batches)
    # logging.debug('state_batches shape: %s', lattice.state_batches.shape)
    # logging.debug('state_batches shape.row_ids(1): %s', lattice.state_batches.shape.row_ids(1))
    logging.debug('state_batches shape.row_ids(2): %s', lattice.state_batches.shape.row_ids(2)) # from this we can get to which time frame that this state belongs to
    logging.debug('state_batches shape.row_ids(2).shape: %s', lattice.state_batches.shape.row_ids(2).shape)


    # by combining the lattice.arcs.row_ids(2) and lattice.state_batches.shape.row_ids(2), we can get the time frame that the arc belongs to

    num_arcs = lattice.num_arcs
    # IndexError: tensors used as indices must be long, byte or bool tensors
    state_index=lattice.arcs.row_ids(2).long()
    time_index=lattice.state_batches.shape.row_ids(2)[state_index][:-1].long() # len=73 for each arc, the last arc is -1, so the time frame is 42
    bpe_index=lattice.arcs.values()[:-1,2].long() # the bpe index of each arc
    logging.debug('time_frame: %s', time_index)
    logging.debug('time_frame shape: %s', time_index.shape)

    # method 1. using hash table to collect the frame and its posterior for each bpe

    bpe_log_post = defaultdict(list) # fir each bpe, we have a list of posterior
    bpe_frame   = defaultdict(list) # for each bpe, we have a list of frame
    # remove the last arc because it is -1
    for i in range(num_arcs-1):
        bpe=int(bpe_index[i])
        bpe_log_post[bpe].append(arc_post[i]) # the posterior is log scale
        t=int(time_index[i])
        bpe_frame[bpe].append(batch['inputs'][0,t,:])
    # print the bpe_posterior if the posterior is not empty
    for bpe in bpe_log_post:
        if len(bpe_log_post[bpe])>0:
            logging.debug('bpe: %s, posterior: %s', bpe, bpe_log_post[bpe])
            logging.debug('bpe: %s, frame: %s', bpe, len(bpe_frame[bpe]))
    # initialize the mu and cov, with shape (K,F)
    mu= torch.zeros((num_bpe, 1024))
    cov=torch.zeros((num_bpe, 1024))

    # re-estimate the parameter of GMM based on the posterior
    for bpe in bpe_log_post:
        if len(bpe_log_post[bpe])>0:
            log_post=torch.stack(bpe_log_post[bpe], dim=0) # posterior is in log scale
            posterior=torch.exp(log_post)
            frame=torch.stack(bpe_frame[bpe], dim=0)
            #mu=torch.sum(posterior.unsqueeze(-1)*frame, dim=0)/torch.sum(posterior)
            mu[bpe]=torch.sum(posterior.unsqueeze(-1)*frame, dim=0)/torch.sum(posterior)
            #cov=torch.sum(posterior.unsqueeze(-1)*(frame-mu)*(frame-mu), dim=0)/torch.sum(posterior)
            cov[bpe]=torch.sum(posterior.unsqueeze(-1)*(frame-mu[bpe])*(frame-mu[bpe]), dim=0)/torch.sum(posterior)
    bpe=123
    logging.debug("mu shape: %s", mu[bpe])
    logging.debug("cov shape: %s", cov[bpe])

    # method 2, store all the log_post to a tensor, with shape (b,T,K)
    # represent, the posterior of each frame at each bpe

    log_post_tensor=torch.log(torch.zeros((batch_size, num_frames, num_bpe)) )
    for i in range(num_arcs-1):
        bpe=int(bpe_index[i])
        t=int(time_index[i] )
        #log_post_tensor[0,t,bpe] = np.logaddexp(log_post_tensor[0,t,bpe], arc_post[i])
        # because there are some duplicated value, like for a frame, there are multiple arcs with the same bpe, so we need to sum them
        log_post_tensor[0,t,bpe] = arc_post[i] if log_post_tensor[0,t,bpe]==-1 else np.logaddexp(log_post_tensor[0,t,bpe], arc_post[i])
        if bpe==123:
            logging.debug('t: %s, arc_post: %s', t, arc_post[i])
    logging.debug('log_post_tensor shape: %s', log_post_tensor[0, :, 123])  # BxTxK
    post_tensor = torch.exp(log_post_tensor)
    logging.debug('post_tensor shape: %s', post_tensor[0, :, 123])  # BxTxK
    # rewrite above for loop using the tensor
    log_post_tensor_2 = torch.log(torch.zeros((batch_size, num_frames, num_bpe)) )
    index=time_index*num_bpe+bpe_index # the absolute index of the tensor time_index*num_bpe+bpe_index
    temp=torch.zeros((num_frames*num_bpe)).double() #T*K
    temp=scatter_add(torch.exp(arc_post[:-1]), index, out=temp)
    temp=torch.log(temp.view(num_frames, num_bpe))
    log_post_tensor_2[0,:,:]=temp

    # check if the two tensors are the same
    logging.debug('two tensors are the same: %s', torch.allclose(log_post_tensor, log_post_tensor_2, atol=1e-5)) # almost the same, may be due to the numerical issue
    logging.debug('temp[bpe=123] %s', temp[:,123])


    # update the mu and cov based on the log_post_tensor

    post_tensor = torch.exp(log_post_tensor_2)


    # shape of batch['inputs']: BxTxF 1,42,1024
    # shape of post_tensor: BxTxK 1,42,500
    # shape of mu and cov: KxF 500,1024
    norm=torch.sum(post_tensor, dim=(0,1)) + 1e-5 # K
    mu_2=torch.sum(post_tensor.unsqueeze(-1)*batch['inputs'].unsqueeze(2), dim=(0,1))/norm.unsqueeze(-1)
    logging.debug('mu_2 shape: %s', mu.shape)
    frame_mu=batch['inputs'].unsqueeze(2)-mu # BxTxKxF
    cov_2=torch.sum(post_tensor.unsqueeze(-1)*(frame_mu*frame_mu), dim=(0,1))/norm.unsqueeze(-1)
    logging.debug('cov_2 shape: %s', cov.shape)

    logging.debug('two tensors are the same: %s', torch.allclose(mu.float(), mu_2.float(), atol=1e-3)) # almost the same, may be due to the numerical issue
    logging.debug('two tensors are the same: %s', torch.allclose(cov.float(), cov_2.float(), atol=1e-5)) # almost the same, may be due to the numerical issue
    bpe=123
    logging.debug("mu shape: %s", mu_2[bpe])
    logging.debug("cov shape: %s", cov_2[bpe])





if __name__ == '__main__':
    # run the main function
    main()
