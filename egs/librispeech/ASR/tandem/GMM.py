
import torch
import torch.nn as nn
from torch.nn import Module
import numpy as np
# from sklearn.mixture import GaussianMixture

# set the seed
torch.manual_seed(0)

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
class GMM_dominik(Module):
    def __init__(self, num_gauss, num_bpe, init_mean, init_std=1, fea_dim=1024):
        super().__init__()

        self.mu = nn.Parameter(
            torch.rand((num_bpe, num_gauss, fea_dim)) * init_std + init_mean.unsqueeze(0).unsqueeze(0),
            requires_grad=False)
        self.log_cov = nn.Parameter(torch.log(torch.ones((num_bpe, num_gauss, fea_dim))), requires_grad=False)
        self.log_pi = nn.Parameter(torch.log(torch.ones((num_bpe, num_gauss)) / num_gauss), requires_grad=False)

    def forward(self, X):
        return compute_gmm_pdf_per_sample(X, self.mu, self.log_cov, self.log_pi)
class Gauss(Module):
    '''
    we implement a diagonal GMM
    '''
    def __init__(self, num_bpe, init_mean, init_std=1, fea_dim=1024):
        super().__init__()

        self.mu = nn.Parameter(
            torch.rand((num_bpe, fea_dim)) * init_std + init_mean,requires_grad=False)

        self.cov = nn.Parameter(torch.ones((num_bpe, fea_dim)), requires_grad=False)
        # self.pi = nn.Parameter(torch.ones((num_gauss) / num_gauss), requires_grad=False)
    def forward(self, x):
        return self.compute_gmm_pdf(x, self.mu, self.cov)
    def compute_gmm_pdf(self, x, mu, cov):
        '''
        x: (batch_size, num_frames, fea_dim)
        mu: (num_bpe, fea_dim)
        cov: (num_bpe, fea_dim)

        return: log_likelihood (batch_size, num_frames, num_bpe)
        '''
        batch_size, num_frames, fea_dim = x.shape
        num_bpe = mu.shape[0]
        assert fea_dim == mu.shape[1]

        # first we implement GMM for each frame
        total_frame= batch_size * num_frames
        x=x.reshape(total_frame,1, fea_dim)
        mu=mu.unsqueeze(0) # (1, num_bpe, fea_dim)
        cov=cov.unsqueeze(0) # (1, num_bpe, fea_dim)
        prec = torch.rsqrt(cov)  # (1, num_bpe, fea_dim)
        prec_squard=prec**2
        log_p = torch.sum((mu * mu + x * x - 2 * x * mu) * prec_squard, dim=2, keepdim=True)
        #log_p = torch.sum((x-mu)**2/cov, dim=2, keepdim=True)
        log_det = torch.sum(torch.log(prec), dim=2, keepdim=True)
        #log_likelihood = -.5 * (fea_dim * np.log(2. * np.pi) + log_p - log_det) #(total_frame,1, num_bpe)
        log_likelihood = -.5 * (fea_dim * np.log(2. * np.pi) + log_p )+ log_det # (total_frame,1, num_bpe)
        return log_likelihood.reshape(batch_size, num_frames, num_bpe)


def test_GMM():
    torch.manual_seed(0)
    # compare the two models
    fea_dim=1
    num_bpe=10
    init_std=1
    model1 = GMM_dominik(1, num_bpe,torch.zeros(num_bpe,1, fea_dim),fea_dim=fea_dim,init_std=init_std)
    model2 = Gauss(num_bpe,torch.zeros(num_bpe, fea_dim),fea_dim=fea_dim,init_std=init_std)

    # compare the forward
    # x = torch.ones(2, 3, fea_dim)
    #x=torch.tensor([[[0],[2.0],[3.0]],[[4.0],[5.0],[6.0]]])
    x=torch.tensor([[[0.0]]])
    y1 = model1(x)
    y2 = model2(x)
    print(y1.shape, y2.shape)
    print("mu's shape", model1.mu.squeeze().shape, model2.mu.squeeze().shape)
    # print the first 10 elements of the two mu's
    # print("mu1", model1.mu.squeeze()[:10])
    # print("mu2", model2.mu.squeeze()[:10])
    print("if the parameters are the same", torch.allclose(model1.mu.squeeze(), model2.mu.squeeze(), atol=1e-6))
    # check the parameters log_cov and cov
    print("log_cov's shape", model1.log_cov.squeeze().shape, model2.cov.squeeze().shape)
    print("if the parameters are the same", torch.allclose(torch.exp(model1.log_cov.squeeze()), model2.cov.squeeze(), atol=1e-6))
    print('the first 10 elements of the two covs')
    # print("log_cov1", torch.exp(model1.log_cov.squeeze()[:10]))
    # print("cov2", model2.cov.squeeze()[:10])
    # print the first 10 elements of the two outputs
    print("y1", y1)
    print("y2", y2)
    # print if the two outputs are the same
    print(torch.allclose(y1, y2, atol=1e-6))

    # use sklearn  GaussianMixture
    # s_gmm=GaussianMixture(n_components=1, covariance_type='diag',init_params='random', random_state=0)
    # #print the mean and cov of the s_gmm
    # print('the mean and cov of the s_gmm', s_gmm.means_, s_gmm.covariances_)






if __name__=='__main__':
    test_GMM()

