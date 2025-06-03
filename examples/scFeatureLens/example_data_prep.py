import torch
import numpy as np
import anndata as ad
import multiDGD
import os
import gc
import math

import sys
sys.path.append(".")
sys.path.append('src')

dev_id = 1
seed = 0
device = torch.device(f"cuda:{dev_id}" if torch.cuda.is_available() else "cpu")
# unzip the data if not already done
data_dir = './data/'
if not os.path.exists(data_dir + 'human_bonemarrow.h5ad'):
    import zipfile
    with zipfile.ZipFile(data_dir + 'human_bonemarrow.h5ad.zip', 'r') as zip_ref:
        zip_ref.extractall(data_dir)
# load the data
data = ad.read_h5ad(data_dir+'human_bonemarrow.h5ad')

model = multiDGD.DGD.load(data=data, save_dir='./data/models/', model_name='human_bonemarrow_l20_h2-3_test50e').to(device)
data = data[data.obs["train_val_test"] == "train"]
library = data.obs['GEX_n_counts'].values
data_gene_names = (data.var[data.var['modality'] == 'GEX']).index
data_gene_ids = data.var[data.var['modality'] == 'GEX']['gene_id'].values
data = data[:,data.var['modality'] != 'ATAC']
data.write_h5ad(data_dir + 'human_bonemarrow_filtered.h5ad')
print("Filtered data saved to:", data_dir + 'human_bonemarrow_filtered.h5ad')
del data
gc.collect()
# get the model's dispersions for the DEG test
with torch.no_grad():
    dispersion_factors = (torch.exp(model.decoder.out_modules[0].distribution.log_r).detach().cpu().numpy() + 1).flatten()

reps = model.representation.z.detach()

# save the reps, the reduced data, and the dispersions
torch.save(reps, data_dir + 'human_bonemarrow_reps.pt')
torch.save(dispersion_factors, data_dir + 'human_bonemarrow_dispersions.pt')
print("Saved representations and dispersion factors.")

###
# calc all sample predictions once here
###
chunk_size = 1000
n_chunks = math.ceil(reps.shape[0] / chunk_size)
y = torch.zeros(reps.shape[0], len(data_gene_names))
for i in range(n_chunks):
    y[i*chunk_size:(i+1)*chunk_size] = model.decoder(torch.cat((reps[i*chunk_size:(i+1)*chunk_size], model.correction_rep.z[i*chunk_size:(i+1)*chunk_size]), dim=1))[0].detach().cpu()
y = y * library.reshape(-1, 1)
y = y.numpy()
# save the predictions
np.save(data_dir + 'human_bonemarrow_predictions.npy', y)
print("Saved predictions for all samples.")