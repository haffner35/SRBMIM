import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import torch
import joblib
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.nn.utils.parametrizations import weight_norm
from siop import get_water_iop
from scipy.io import loadmat


class ProjectionNet(nn.Module):
    def __init__(self, input_dim=6, hidden_dims=(4, 2)):
        super().__init__()
        self.fc1 = weight_norm(nn.Linear(input_dim, hidden_dims[0]))
        self.act = nn.LeakyReLU(inplace=True)
        self.fc2 = weight_norm(nn.Linear(hidden_dims[0], hidden_dims[1]))
    def forward(self, z):
        z = self.fc1(z)
        z = self.act(z)
        z = self.fc2(z)
        return z



saved_models = joblib.load("residual_covariance_opt_demo.joblib")
saved_data = joblib.load("soft_reconstruction_demo.joblib")
pure_watar_iop = joblib.load("pure_water_iop.joblib")

bbp_aop = saved_data["bbp_aop"]
aph_aop = saved_data["aph_aop"]
adg_aop = saved_data["adg_aop"]
bands = saved_data["wavelength"].copy()

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

split_ratio = 0.2
num_iter = 10000

epsilon_reg = 1e-3
softplus_reg = 1e-8
train_in_log = True
log_eps = 1e-12

use_cov_reg = True
lambda_cov = 10
sigmoid_tau = 0.1
G_grid = torch.linspace(0.5, 3.0, 24)

aw = pure_watar_iop["aw"]
bbw = pure_watar_iop["bbw"]





g0 = 0.0895
g1 = 0.1247

paras = ["bbp", "aph", "adg"]
paras2 = ["bbp", "aph", "adg"]

inv_results = {}

for para in paras:

    if para == "bbp":
        iop_matrix = bbp_aop.copy()
        lower_bound = 1e-4
        upper_bound = 0.5
        para_loc = 0
    elif para == "aph":
        iop_matrix = aph_aop.copy()
        lower_bound = 1e-3
        upper_bound = 1
        para_loc = 1
    else:
        iop_matrix = adg_aop.copy()
        lower_bound = 1e-3
        upper_bound = 5
        para_loc = 2


    Rrs = iop_matrix.filter(like='Rrse').values
    pixel_num, band_num = Rrs.shape



    rrs = Rrs / (0.52 + 1.7 * Rrs)
    x_true = iop_matrix[f"{para}443"].values

# soft reconstruction
    beta_dict = {}
    for para2 in paras2:

        M_beta = np.zeros((pixel_num, band_num))
        for j, band in enumerate(bands):

            sr_para = saved_models[(para2, band)]

            net = ProjectionNet(input_dim=band_num)
            net.load_state_dict(sr_para["model_state"])
            net.eval()
            centers = torch.tensor(sr_para["centers"], dtype=torch.float32)
            betas = torch.tensor(sr_para["betas"], dtype=torch.float32)
            scaler_z = sr_para["scaler_z"]
            tau = sr_para["current_tau"]

            Rrs_std = scaler_z.transform(Rrs)
            Rrs_tensor = torch.tensor(Rrs_std, dtype=torch.float32)

            with torch.no_grad():
                q = net(Rrs_tensor)
                dist_sq = ((q.unsqueeze(1) - centers.unsqueeze(0)) ** 2).sum(dim=2)
                u = torch.softmax(-dist_sq / tau, dim=1)

            betas = betas.view(betas.shape[0], -1).squeeze(1)
            M_beta[:, j] = (u * betas.unsqueeze(0)).sum(dim=1).cpu().numpy()

        beta_dict[para2] = M_beta

    sbbp = beta_dict["bbp"]
    saph = beta_dict["aph"]
    sadg = beta_dict["adg"]

# build the basis matrix
    A_list, b_list = [], []
    for j in range(pixel_num):
        u_j = (-g0 + np.sqrt(g0 ** 2 + 4 * g1 * rrs[j])) / (2 * g1)
        A_j = np.stack([(u_j - 1) * sbbp[j], u_j * saph[j], u_j * sadg[j]], axis=1)
        b_j = (1 - u_j) * bbw - u_j * aw
        A_list.append(A_j)
        b_list.append(b_j)

    idx_all = np.arange(pixel_num)
    train_idx, test_idx = train_test_split(
        idx_all, train_size=split_ratio, shuffle=True, random_state=seed
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    G_grid = G_grid.to(device)

    A_full = torch.stack([torch.as_tensor(A_list[i], dtype=dtype, device=device) for i in range(pixel_num)])
    b_full = torch.stack([torch.as_tensor(b_list[i], dtype=dtype, device=device) for i in range(pixel_num)])
    x_true_tensor = torch.as_tensor(x_true, dtype=dtype, device=device)

    A = A_full[train_idx]
    b = b_full[train_idx]
    x_obs = x_true_tensor[train_idx]
    x_test_obs = x_true_tensor[test_idx]


    tril_idx = torch.tril_indices(band_num, band_num, device=device)
    I3 = torch.eye(3, dtype=dtype, device=device)

    L_vec = torch.zeros(len(tril_idx[0]), dtype=dtype, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([L_vec], lr=1e-3)


    for iter in range(num_iter):

        optimizer.zero_grad()

        L_mat = torch.zeros(band_num, band_num, dtype=dtype, device=device)
        L_mat[tril_idx[0], tril_idx[1]] = L_vec
        d = torch.arange(band_num, device=device)
        L_mat[d, d] = F.softplus(L_mat[d, d]) + softplus_reg
        W = L_mat @ L_mat.T

        Q = torch.einsum("bji,jk->bik", A, W)
        Q = torch.einsum("bij,bjk->bik", Q, A)
        g = torch.einsum("bji,jk->bik", A, W)
        g = torch.einsum("bij,bj->bi", g, b)
        x_pred = torch.linalg.solve(Q + epsilon_reg * I3, g)

        if train_in_log:
            eps = 1e-4
            x_pred_para = torch.clamp(x_pred[:, para_loc], min=eps)
            loss_pred = torch.abs(torch.log10(x_pred_para / x_obs)).mean()
        else:
            eps = 1e-4
            x_pred_para = torch.clamp(x_pred[:, para_loc], min=eps)
            loss_pred = torch.mean((x_obs - x_pred_para) ** 2)

        # coverage regularization
        cov_reg = torch.tensor(0, dtype=dtype, device=device)
        if use_cov_reg:
            Q_reg = Q + epsilon_reg * I3
            Q_reg_inv = torch.linalg.inv(Q_reg)

            Ax_pred = torch.einsum("bij,bj->bi", A, x_pred)
            r = b - Ax_pred
            rss = torch.einsum("bi,ij,bj->b", r, W, r)

            dof = bands.shape[0] - 3
            sigma2 = (rss / dof).unsqueeze(-1).unsqueeze(-1)
            Sigma = sigma2 * Q_reg_inv

            var_para = Sigma[:, para_loc, para_loc].clamp_min(1e-25)
            std_para = torch.sqrt(var_para)

            if train_in_log:
                eps = 1e-4
                x_pred_para = torch.clamp(x_pred[:, para_loc], min=eps)
                log10 = np.log(10)
                log10_err = torch.log10(x_obs / x_pred_para)
                sigma_log10 = std_para / (x_pred_para * log10)
                z = log10_err / sigma_log10
            else:
                eps = 1e-4
                x_pred_para = torch.clamp(x_pred[:, para_loc], min=eps)
                z = (x_obs - x_pred_para) / std_para

            abs_z = torch.abs(z).unsqueeze(1)
            gs = G_grid.view(1, -1)
            soft_hit = torch.sigmoid((gs - abs_z) / sigmoid_tau)
            emp_cov = soft_hit.mean(dim=0)
            gaussian_cov = torch.special.ndtr(G_grid) - torch.special.ndtr(-G_grid)
            cov_reg = torch.mean((emp_cov - gaussian_cov) ** 2)

        loss = loss_pred + lambda_cov * cov_reg
        loss.backward()
        optimizer.step()

        if (iter % 2000) == 0:
            print(f"[{para}] iter={iter:5d} loss={loss.item():.4e}")


    with torch.no_grad():
        L_mat = torch.zeros(band_num, band_num, dtype=dtype, device=device)
        L_mat[tril_idx[0], tril_idx[1]] = L_vec
        d = torch.arange(band_num, device=device)
        L_mat[d, d] = F.softplus(L_mat[d, d]) + softplus_reg
        W = L_mat @ L_mat.T

        Q_test = torch.einsum("bji,jk->bik", A_full[test_idx], W)
        Q_test = torch.einsum("bij,bjk->bik", Q_test, A_full[test_idx])
        g_test = torch.einsum("bji,jk->bik", A_full[test_idx], W)
        g_test = torch.einsum("bij,bj->bi", g_test, b_full[test_idx])

        x_inv_tensor = torch.linalg.solve(Q_test + epsilon_reg * I3, g_test)

        r_test = b_full[test_idx] - torch.einsum("bij,bj->bi", A_full[test_idx], x_inv_tensor)
        rss_test = torch.einsum("bi,ij,bj->b", r_test, W, r_test)

        S_eps = Q_test + epsilon_reg * I3
        S_eps_inv = torch.linalg.inv(S_eps)

        sigma2_test = (rss_test / dof).unsqueeze(-1).unsqueeze(-1)
        Sigma_test = sigma2_test * S_eps_inv

        var_test = Sigma_test[:, para_loc, para_loc].clamp_min(1e-25)
        std_test = torch.sqrt(var_test)

        x_inv = x_inv_tensor.cpu().numpy()
        x_inv_std = std_test.cpu().numpy()

    x_inv = np.clip(x_inv, lower_bound, upper_bound)
    inv_results[para] = x_inv[:, para_loc]
    inv_results[para + "_unc"] = x_inv_std


print("All inversions completed.")



