import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os, argparse, time
from scipy import sparse
from tch.model import SinkhornVAE, SinkhornSoftK
from tch.sinkhorn import CELoss, getSinkhornLoss
from utils.util import dataloader, load_train_data, load_tr_te_data, ndcg_at_k

parser = argparse.ArgumentParser(description="")
parser.add_argument("-dataset", default="ml-20m")
parser.add_argument("-batch_size", type=int, default=500)
parser.add_argument("-n_epochs", type=int, default=100)

args = parser.parse_args()

dataset   = args.dataset
data_dir  = "/home/xx/data/{}".format(dataset)
chkpt_dir = "/tmp/torch/sinkhorncf/{}/large".format(dataset)
use_cuda = True
device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")
print(device)

torch.manual_seed(3)
np.random.seed(3)

################################################################################

pro_dir = os.path.join(data_dir, 'large')
if not os.path.isdir(chkpt_dir):
    os.makedirs(chkpt_dir)

unique_sid = np.genfromtxt(os.path.join(pro_dir, 'unique_sid.txt'), dtype='str')
n_items = len(unique_sid)
print("Number of items: {}".format(n_items))

train_data = load_train_data(os.path.join(pro_dir, 'train.csv'), n_items)
#soft_train_data = load_train_data(os.path.join(pro_dir, 'soft-train.csv'), n_items)
vald_data_tr, vald_data_te = load_tr_te_data(os.path.join(pro_dir, 'validation_tr.csv'),
                                             os.path.join(pro_dir, 'validation_te.csv'),
                                             n_items)
## pre-computed cost matrix between items
C = np.load(os.path.join(pro_dir, 'cm.npy'))
SinkhornLoss = getSinkhornLoss(C, 1.0, 5)

N = train_data.shape[0]
idx = np.arange(N)
n_epochs = args.n_epochs
batch_size = args.batch_size

N_vald = vald_data_tr.shape[0]
batch_size_vald = 2000

total_anneal_steps = 200000
anneal_cap_table = {'ml-20m': 0.2, 'millionsong': 0.2, 'netflix': 0.1}
anneal_cap = anneal_cap_table[dataset]
dropout_prob = 0.5
p_dims = [200, 600, n_items]
α_table = {'ml-20m': 0.03, 'millionsong': 0.002, 'netflix': 0.01}
α = α_table[dataset]

print("Start training...")

vae = SinkhornVAE(p_dims, dropout_prob).to(device)
#vae = SinkhornSoftK(p_dims, dropout_prob, 300).to(device)
vae.set_prior(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3, amsgrad=True)
loaddata_train = dataloader(train_data)
loaddata_valad = dataloader(vald_data_tr, vald_data_te)

def train_step(epoch, i, X, Y, β):
    logits, KLD = vae(X)
    ceLoss = CELoss(Y, logits)
    skLoss = SinkhornLoss(X, logits, k=100) if epoch > 10 and i % 100 == 0 else 0.0
    loss = ceLoss + skLoss + β * KLD
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return ceLoss, skLoss

best_ndcg = -np.inf
update_count = 0
tic = time.time()
for epoch in range(n_epochs):
    np.random.shuffle(idx)
    for st_idx in range(0, N, batch_size):
        end_idx = min(st_idx + batch_size, N)
        X = loaddata_train(idx[st_idx:end_idx])
        X = torch.FloatTensor(X).to(device)
        Y = torch.where(X < 1.0, α*torch.ones_like(X), X)

        β = min(anneal_cap, update_count / total_anneal_steps)
        ceLoss, skLoss = train_step(epoch, update_count, X, Y, β)
        update_count += 1

    #print("Epoch {}: ceLoss {:.5f}, skLoss {:.5f}".format(epoch, ceLoss, skLoss))
    ndcg_list = []
    vae.eval()
    for st_idx in range(0, N_vald, batch_size_vald):
        end_idx = min(st_idx + batch_size_vald, N_vald)
        X_tr, X_te = loaddata_valad(np.arange(st_idx, end_idx))
        ## infer from partial observation X_tr
        with torch.no_grad():
            logits, _ = vae(torch.FloatTensor(X_tr).to(device))
        logits = logits.cpu().numpy()
        logits[X_tr.nonzero()] = -np.inf
        ## evaluate the inferred results with observation X_te
        ndcg_list.append(ndcg_at_k(logits, X_te, k=100))
    vae.train()
    ndcg_list = np.concatenate(ndcg_list)
    ndcg = np.mean(ndcg_list)
    if ndcg > best_ndcg:
        torch.save(vae.state_dict(), os.path.join(chkpt_dir, 'model.pt'))
        best_ndcg = ndcg
        print('Epoch: {} Best NDCG:{:.5f}'.format(epoch, ndcg))
print("{} minutes passed".format((time.time() - tic) / 60.))