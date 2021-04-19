import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributions as dist
import numpy as np
from scipy import sparse
from pyro.distributions.transforms import affine_autoregressive

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 dropout_prob=0.0, nonlinear_f=None):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        if nonlinear_f == 'selu':
            self.nonlinear_f = F.selu
        elif nonlinear_f == 'tanh':
            self.nonlinear_f = torch.tanh
        elif nonlinear_f == 'leakyrelu':
            self.nonlinear_f = F.leaky_relu
        else:
            self.nonlinear_f = F.relu

        self.dropout = nn.Dropout(dropout_prob)
    def forward(self, x):
        h1 = self.dropout(self.nonlinear_f(self.fc1(x)))
        return self.fc2(h1)
        

class Reverse(nn.Module):
    """
    reverse = Reverse(5)
    u = torch.rand(10, 5)
    x = reverse.forward(u)
    reverse.inverse(x)
    """
    def __init__(self, D):
        super(Reverse, self).__init__()
        self.perm = np.arange(D-1, -1, -1)
        self.inv_perm = self.perm

    def forward(self, u):
        return u[:, self.inv_perm]

    def inverse(self, x):
        return x[:, self.perm]

    def log_abs_det_jacobian(self, u, x):
        return torch.zeros(x.size()[:-1], dtype=x.dtype, device=x.device, layout=x.layout)
        
class SinkhornVAE(nn.Module):
    def __init__(self, p_dims, dropout_prob):
        super(SinkhornVAE, self).__init__()
        self.p_dims = p_dims
        self.q_dims = p_dims[::-1]

        self.dropout = nn.Dropout(dropout_prob)
        self.q = MLP(self.q_dims[0], self.q_dims[1], 2*self.q_dims[2])
        self.p = MLP(self.p_dims[0], self.p_dims[1], self.p_dims[2])
        self.q.apply(init_linear)
        self.p.apply(init_linear)
        transforms = createTransforms(2, p_dims[0])
        self.transforms = nn.ModuleList(transforms)
        self.prior = None

    def forward(self, x):
        """
        Args:
            x (batch_size, m)
        Returns:
            logits (batch_size, m)
        """
        h = F.normalize(x, dim=1)
        h = self.dropout(h)
        
        o = self.q(h)
        μ, logσ2 = torch.split(o, o.shape[1]//2, dim=1)
        σ = torch.exp(0.5*logσ2)
        z, KLD = posteriorKLD(μ, σ, self.transforms, self.prior, self.training)
        logits = self.p(z)

        return logits, KLD

    def set_prior(self, device):
        self.prior = normal_prior(self.p_dims[0], device)

def init_linear(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


def normal_prior(dim, device=torch.device("cpu")):
    """
    Return Normal(0, 1) of dimension `dim`.
    """
    normal = dist.Normal(torch.zeros(dim, device=device), torch.ones(dim, device=device))
    return dist.Independent(normal, reinterpreted_batch_ndims=1)
   

def posteriorKLD(μ, σ, transforms, prior, training):
    """
    Return posterior samples and KL Divergence from posterior to prior distribution.
    """
    base_dist = dist.Independent(dist.Normal(loc=μ, scale=σ), 
                                 reinterpreted_batch_ndims=1)
    def reparameterize(μ, σ):
        if training:
            ϵ = torch.randn_like(σ)
            return μ + σ * ϵ
        else:
            return μ
    def transform_and_log_prob(z):
        u = z
        J = 0.0
        for transform in transforms:
            x = transform(u)
            J = J + transform.log_abs_det_jacobian(u, x)
            u = x
        log_prob = base_dist.log_prob(z) - J
        return u, log_prob
    # z = reparameterize(μ, σ)
    # KLD = torch.mean(-0.5 * torch.sum(1 + 2*torch.log(σ) - σ**2 - μ**2, dim=1))
    # return z, KLD
    z = reparameterize(μ, σ)
    z, log_q = transform_and_log_prob(z)
    KLD = torch.mean(log_q - prior.log_prob(z))
    return z, KLD

def createTransforms(n, dim):
    transforms = []
    for _ in range(n):
        transforms += [affine_autoregressive(dim), Reverse(dim)]
    return transforms[:-1] if len(transforms) > 0 else transforms


class SinkhornSoftK(nn.Module):
    def __init__(self, p_dims, dropout_prob, K):
        super(SinkhornSoftK, self).__init__()
        self.p_dims = p_dims
        self.q_dims = p_dims[::-1]
        self.z_dim = self.q_dims[2]

        self.dropout = nn.Dropout(dropout_prob)
        self.q = MLP(self.q_dims[0], self.q_dims[1], K)
        self.p = MLP(self.p_dims[0], self.p_dims[1], self.p_dims[2])
        self.q.apply(init_linear)
        self.p.apply(init_linear)

        self.k_embedding = nn.Embedding(K, self.z_dim)

    def forward(self, x):
        """
        Args:
            x (batch_size, m)
        Returns:
            logits (batch_size, m)
        """
        h = F.normalize(x, dim=1)
        h = self.dropout(h)
        
        o = self.q(h)
        z = self.soft_representation(o)
        logits = self.p(z)

        return logits, 0

    def soft_representation(self, o):
        """
        Args:
            o (batch, K)
        Returns:
            z (batch, z_dim)
        """
        π = torch.softmax(o, dim=1)
        return torch.mm(π, self.k_embedding.weight)
    
    def set_prior(self, device):
        pass
