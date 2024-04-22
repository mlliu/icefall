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
    #gmm_layer = GMM(10, num_bpe,torch.zeros(num_bpe, 10, 1024))

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
    sampler = SimpleCutSampler(cs, shuffle=False, max_cuts=10) # batch size is 1
    loader = DataLoader(dataset, sampler=sampler, batch_size=None, num_workers=4, prefetch_factor=4)

    batch = next(iter(loader))
    # use the log to print the batch
    logging.debug('batch supervision: %s', batch['supervisions']['text'] )
    logging.debug('batch supervision num_frames: %s', batch['supervisions']['num_frames'])
    logging.debug('batch inputs: %s', batch['inputs'].shape)

    net_out = gmm_layer(batch['inputs'].to(device)) # BxTxK
    logging.info('successfully forward the batch through the GMM layer')
    batch_size= net_out.shape[0]
    num_frames= net_out.shape[1]



    logging.debug('net_out: %s', net_out.shape)
    supervision_segments, texts = encode_supervisions(batch['supervisions'], 1)
    logging.debug('supervision_segments: %s', supervision_segments) # [[0,0,111],[1,0,42]] when batch_size=2
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
    logging.debug('hmm_graph shape: %s and length of the shape: %s', hmm_graph.shape, len(hmm_graph.shape)) # hmm_graph shape (2, None, None) when batch_size=2




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
    lattice_fsa_1= lattice[0]
    # lattice_fsa.labels_sym = graph.word_table
    # lattice_fsa.aux_labels_sym = graph.word_table
    #lattice_fsa_1.draw('graph/lattice_batch1_beam100.pdf', title='Lattice with beam 100')
    logging.debug('lattice 1 num_arcs: %s', lattice_fsa_1.num_arcs)

    lattice_fsa_2= lattice[1]
    # lattice_fsa.labels_sym = graph.word_table
    # lattice_fsa.aux_labels_sym = graph.word_table
    #lattice_fsa_2.draw('graph/lattice_batch2_beam100.pdf', title='Lattice with beam 100')
    logging.debug('lattice 2 num_arcs: %s', lattice_fsa_2.num_arcs)


    # nll = -(lattice.get_tot_scores(log_semiring=True, use_double_scores=True) / batch['supervisions'][
    #     'num_frames'].to(device)).sum()
    tot_score = lattice._get_tot_scores(log_semiring=True, use_double_scores=True)
    forward_score = lattice._get_forward_scores(log_semiring=True, use_double_scores=True)
    backward_score = lattice._get_backward_scores(log_semiring=True, use_double_scores=True)

    #valid_arcs,there are total num_arcs=508 arcs, but only those whose values are not -1 are valid, coolect the valid arcs index
    arc_values  = lattice.arcs.values()
    valid_arcs_mask = arc_values[:, 2] != -1
    num_valid_arcs = int(torch.sum(valid_arcs_mask))

    logging.debug('num_valid_arcs: %s', num_valid_arcs) # 504
    # logging.debug('valid_arcs_mask: %s', valid_arcs_mask)
    logging.debug('valid_arcs_mask shape: %s', valid_arcs_mask.shape) # shape should be 508-4

    arc_post = lattice._get_arc_post(log_semiring=True, use_double_scores=True)  # 74
    # use the valid_arcs_mask to filter the arc_post
    arc_post=arc_post[valid_arcs_mask]
    num_arcs=lattice.num_arcs
    logging.debug('tot_score: %s', tot_score)
    logging.debug('forward_score: %s', forward_score.shape) # len=404 for each state
    # logging.debug('forward_score: %s', forward_score)
    logging.debug('backward_score: %s', backward_score.shape)# len=404 for each state
    # logging.debug('backward_score: %s', backward_score)
    # logging.debug('arc_post: %s', arc_post)
    logging.debug('arc_post. shape: %s', arc_post.shape) # len=508 for each arc
    logging.debug('num_arcs: %s', num_arcs) # 508
    logging.debug('arcs.values(): %s', lattice.arcs.values()) # (source state, dest state, label, score_int) len=74
    logging.debug('arcs.values().shape: %s', lattice.arcs.values().shape) # (508, 4)
    # logging.debug('scores: %s', lattice.scores)
    #logging.debug('labels: %s', lattice.labels) # the labels of the arcs is
    # logging.debug('properties: %s', lattice.properties)  # the labels of the arcs is
    # logging.debug('row_splits: %s', lattice[0].shape)
    # logging.debug('row_splits(1): %s', lattice.arcs.row_splits(1))
    # logging.debug('row_splits(2): %s', lattice.arcs.row_splits(2))
    # logging.debug('row_ids(1): %s', lattice.arcs.row_ids(1)) # which batch/fsa that this state belongs to
    # logging.debug('row_ids(1).shape: %s', lattice.arcs.row_ids(1).shape) # 404
    # logging.debug('row_ids(2): %s', lattice.arcs.row_ids(2)) # from this we can get to which state that this arc belongs to
    # logging.debug('row_ids(2).shape: %s', lattice.arcs.row_ids(2).shape) # 508



    # by combining the lattice.arcs.row_ids(2) and lattice.arcs.row_ids(1), we can get to which batch that this arc belongs to
    state_index=lattice.arcs.row_ids(2).long() # to which state that this arc belongs to

    batch_index=lattice.arcs.row_ids(1)[state_index].long() # to which batch that this arc belongs to
    # use the valid_arcs_mask to filter the batch_index
    batch_index=batch_index[valid_arcs_mask]

    logging.debug('state_index: %s', state_index)
    logging.debug('state_index shape: %s', state_index.shape)
    logging.debug('batch_index: %s', batch_index)
    # count how many 1s in the batch_index
    logging.debug('1 in batch_index: the number of arcs that belongs to batch 2 %s', torch.sum(batch_index==1)) #98
    logging.debug('batch_index shape: %s', batch_index.shape)


    # logging.debug('state_batches: %s', lattice.state_batches)
    # logging.debug('state_batches value: %s', lattice.state_batches.values)
    # logging.debug('state_batches value shape: %s', lattice.state_batches.values.shape) # 404
    # logging.debug('state_batches shape.row_splits(1): %s', lattice.state_batches.shape.row_splits(1)) # the start index of each batch
    # logging.debug('state_batches shape.row_splits(1).shape: %s', lattice.state_batches.shape.row_splits(1).shape)
    # logging.debug('state_batches shape.row_splits(2): %s', lattice.state_batches.shape.row_splits(2)) # the start index of each fsa
    # logging.debug('state_batches shape.row_splits(2).shape: %s', lattice.state_batches.shape.row_splits(2).shape)
    # logging.debug('state_batches shape.row_ids(1): %s', lattice.state_batches.shape.row_ids(1)) #
    # logging.debug('state_batches shape.row_ids(1).shape: %s', lattice.state_batches.shape.row_ids(1).shape)
    # logging.debug('state_batches shape.row_ids(2): %s', lattice.state_batches.shape.row_ids(2)) # from this we can get to which time frame that this state belongs to
    # logging.debug('state_batches shape.row_ids(2).shape: %s', lattice.state_batches.shape.row_ids(2).shape)

    #  by combining the lattice.state_batches.shape.row_ids(2) and lattice.state_batches.shape.row_ids(1), we can get to time frame that this state belongs to
    # this is not working, because the two fsas are mixed together
    state_rowid2=lattice.state_batches.shape.row_ids(2).long()
    state_time_frame=lattice.state_batches.shape.row_ids(1)[state_rowid2].long() # to which time frame that this state belongs to
    flat_values=lattice.state_batches.values.long() # the corresponding state index

    logging.debug('state_time_frame: %s', state_time_frame)
    logging.debug('state_time_frame shape: %s', state_time_frame.shape)
    logging.debug('flat_values: %s', flat_values)
    logging.debug('flat_values shape: %s', flat_values.shape)

    # initialize the time_index, with the shape same as state_time_frame
    state_time_index=torch.ones_like(state_time_frame)
    # reorgnize the state_time_frame based on the flat_value
    state_time_index[flat_values]=state_time_frame # to which time frame that this state belongs to
    logging.debug('state_time_index: %s', state_time_index)
    logging.debug('state_time_index shape: %s', state_time_index.shape)

    time_index = state_time_index[state_index] # from which time frame that this arc belongs to
    # use the valid_arcs_mask to filter the time_index
    time_index=time_index[valid_arcs_mask]
    logging.debug('time_index: %s', time_index)
    logging.debug('time_index shape: %s', time_index.shape)
    bpe_index = lattice.arcs.values()[:, 2].long() # the bpe index of each arc
    # use the valid_arcs_mask to filter the bpe_index
    bpe_index=bpe_index[valid_arcs_mask]
    logging.debug('bpe_index: %s', bpe_index)
    logging.debug('bpe_index shape: %s', bpe_index.shape)




    # by combining the lattice.arcs.row_ids(2) and lattice.state_batches.shape.row_ids(2), we can get the time frame that the arc belongs to


    # # IndexError: tensors used as indices must be long, byte or bool tensors
    # state_index=lattice.arcs.row_ids(2).long()
    # time_index=lattice.state_batches.shape.row_ids(2)[state_index][:-1].long() # len=73 for each arc, the last arc is -1, so the time frame is 42
    # bpe_index=lattice.arcs.values()[:-1,2].long() # the bpe index of each arc
    #         logging.debug('time_frame: %s', time_index)
    #         logging.debug('time_frame shape: %s', time_index.shape) # 507

    # method 1. using hash table to collect the frame and its posterior for each bpe

    bpe_log_post = defaultdict(list) # fir each bpe, we have a list of posterior
    bpe_frame   = defaultdict(list) # for each bpe, we have a list of frame
    # remove the last arc because it is -1
    for i in range(num_valid_arcs):
        bpe=int(bpe_index[i])
        bpe_log_post[bpe].append(arc_post[i]) # the posterior is log scale
        b_index= int(batch_index[i])
        t=int(time_index[i])
        bpe_frame[bpe].append(batch['inputs'][b_index,t,:])
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
    bpe=88
    logging.debug("mu shape: %s", mu[bpe])
    logging.debug("cov shape: %s", cov[bpe])

    # method 2, store all the log_post to a tensor, with shape (b,T,K)
    # represent, the posterior of each frame at each bpe

    log_post_tensor=torch.log(torch.zeros((batch_size, num_frames, num_bpe)) )
    logging.debug('num_valid_arcs: %s', num_valid_arcs) # 504
    count_88=0
    for i in range(num_valid_arcs):
        bpe=int(bpe_index[i])
        t=int(time_index[i] )
        b_index= int(batch_index[i])
        #log_post_tensor[0,t,bpe] = np.logaddexp(log_post_tensor[0,t,bpe], arc_post[i])
        # because there are some duplicated value, like for a frame, there are multiple arcs with the same bpe, so we need to sum them
        log_post_tensor[b_index,t,bpe] = arc_post[i] if log_post_tensor[b_index,t,bpe]==-1 else np.logaddexp(log_post_tensor[b_index,t,bpe], arc_post[i])
        if bpe==88:
            count_88+=1
            logging.debug('i %s, b_index: %s, t: %s, bpe: %s, arc_post: %s', i, b_index, t, bpe, arc_post[i])
        if i==502:
            logging.debug('i %s, b_index: %s, t: %s, bpe: %s, arc_post: %s', i, b_index, t, bpe, arc_post[i])
        if i==503:
            logging.debug('i %s, b_index: %s, t: %s, bpe: %s, arc_post: %s', i, b_index, t, bpe, arc_post[i])
            #logging.debug('t: %s, arc_post: %s', t, arc_post[i])
    logging.debug('count_88: %s', count_88)
    logging.debug('log_post_tensor shape: %s', log_post_tensor[1, :, 88])  # BxTxK
    post_tensor = torch.exp(log_post_tensor)
    logging.debug('post_tensor shape: %s', post_tensor[1, :, 88])  # BxTxK
    # rewrite above for loop using the tensor
    log_post_tensor_2 = torch.log(torch.zeros((batch_size, num_frames, num_bpe)) )
    # index=time_index*num_bpe+bpe_index # the absolute index of the tensor time_index*num_bpe+bpe_index
    # temp=torch.zeros((num_frames*num_bpe)).double() #T*K
    # temp=scatter_add(torch.exp(arc_post[:-1]), index, out=temp)
    # temp=torch.log(temp.view(num_frames, num_bpe))
    # log_post_tensor_2[0,:,:]=temp
    index=batch_index*num_frames*num_bpe+time_index*num_bpe+bpe_index # the absolute index of the tensor time_index*num_bpe+bpe_index
    temp=torch.zeros((batch_size*num_frames*num_bpe)).double() #T*K
    temp=scatter_add(torch.exp(arc_post), index, out=temp)
    temp=torch.log(temp.view(batch_size, num_frames, num_bpe))
    log_post_tensor_2=temp

    # check if the two tensors are the same
    logging.debug('two tensors are the same: %s', torch.allclose(log_post_tensor.float(), log_post_tensor_2.float(), atol=1e-3))
    # check from which batch, which bpe, the two tensors are different
    for i in range(batch_size):
            for k in range(num_bpe):
                if not torch.allclose(log_post_tensor[i,:,k].float(), log_post_tensor_2[i,:,k].float(), atol=1e-3):
                    logging.debug('where is the dfference from: batch: %s, bpe: %s, log_post_tensor: %s, log_post_tensor_2: %s', i, k, log_post_tensor[i,:,k], log_post_tensor_2[i,:,k])

    # check how many bpe=88 in the bpe_index
    logging.debug('88 in bpe_index: the number of arcs that has to bpe 88 %s', torch.sum(bpe_index==88)) # 42

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

    logging.debug('two tensors are the same: %s', torch.allclose(mu.float(), mu_2.float(), atol=1e-3))
    # check for which bpe
    for i in range(num_bpe):
        if not torch.allclose(mu[i].float(), mu_2[i].float(), atol=1e-3):
            logging.debug('where is the dfference from: bpe: %s, mu: %s, mu_2: %s', i, mu[i,:], mu_2[i,:])

    bpe=88
    logging.debug("mu shape: %s", mu_2[bpe])
    logging.debug("cov shape: %s", cov_2[bpe])
    logging.info ("mu and cov are updated correctly")





if __name__ == '__main__':
    # run the main function
    main()
