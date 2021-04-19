import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os, argparse
from scipy import sparse
from collections import defaultdict
from tch.model import  SinkhornVAE, SinkhornSoftK
from utils.util import dataloader, load_tr_te_data, ndcg_at_k, recall_at_k

parser = argparse.ArgumentParser(description="")
parser.add_argument("-dataset", default="ml-20m")

args = parser.parse_args()

dataset   = args.dataset
data_dir  = "/home/xx/data/{}".format(dataset)
chkpt_dir = "/tmp/torch/sinkhorncf/{}/large".format(dataset)
use_cuda = True
device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
print(device)

pro_dir = os.path.join(data_dir, 'large')
unique_sid = np.genfromtxt(os.path.join(pro_dir, 'unique_sid.txt'), dtype='str')
n_items = len(unique_sid)

test_data_tr, test_data_te = load_tr_te_data(os.path.join(pro_dir, 'test_tr.csv'),
                                             os.path.join(pro_dir, 'test_te.csv'),
                                             n_items)

p_dims = [200, 600, n_items]
N_test = test_data_tr.shape[0]
batch_size = 2000
print("Start testing...")
vae = SinkhornVAE(p_dims, 0.0).to(device)
#vae = SinkhornSoftK(p_dims, 0.0, 300).to(device)
vae.load_state_dict(torch.load(os.path.join(chkpt_dir, 'model.pt')))
vae.set_prior(device)
vae.eval()
loaddata = dataloader(test_data_tr, test_data_te)

n_dict = {5: "nDCG@5", 10: "nDCG@10", 15: "nDCG@15", 20: "nDCG@20",
	25: "nDCG@25", 50: "nDCG@50", 75: "nDCG@75", 100: "nDCG@100"}
r_dict = {5: "Recall@5", 10: "Recall@10", 15: "Recall@15", 20: "Recall@20",
    25: "Recall@25", 50: "Recall@50", 75: "Recall@75", 100: "Recall@100"}

n_list = defaultdict(list)
r_list = defaultdict(list)

for st_idx in range(0, N_test, batch_size):
    end_idx = min(st_idx + batch_size, N_test)
    X_tr, X_te = loaddata(np.arange(st_idx, end_idx))

    with torch.no_grad():
        logits, _ = vae(torch.FloatTensor(X_tr).to(device))
    ## move to cpu
    logits = logits.cpu().numpy()
    logits[X_tr.nonzero()] = -np.inf

    for n_key in n_dict.keys():
        n_list[n_key].append(ndcg_at_k(logits, X_te, k=n_key))

    for r_key in r_dict.keys():
        r_list[r_key].append(recall_at_k(logits, X_te, k=r_key))

for n_key in n_dict.keys():
    n_list[n_key] = np.concatenate(n_list[n_key])
for r_key in r_dict.keys():
    r_list[r_key] = np.concatenate(r_list[r_key])

N = np.sqrt(len(n_list[5]))
for n_key in sorted(n_dict.keys()):
    print("{:15s} = {:.5f} ({:.5f})".format("Test {}".format(n_dict[n_key]), np.mean(n_list[n_key]), np.std(n_list[n_key]) / N))
for r_key in sorted(r_dict.keys()):
    print("{:15s} = {:.5f} ({:.5f})".format("Test {}".format(r_dict[r_key]), np.mean(r_list[r_key]), np.std(r_list[r_key]) / N))