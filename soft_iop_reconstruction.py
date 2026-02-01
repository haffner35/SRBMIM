import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


# map the original Rrs to low-dimensional subspace
class ProjectionNet(nn.Module):
    def __init__(self, input_dim=6, hidden_dims=(4, 2)):
        super().__init__()
        self.fc1 = weight_norm(nn.Linear(input_dim, hidden_dims[0]))
        self.act = nn.LeakyReLU(inplace=True)
        self.fc2 = weight_norm(nn.Linear(hidden_dims[0], hidden_dims[1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


def soft_clustering_regression(
    x_tensor,     # iop at 443 nm
    y_tensor,  # iop at target wavelength
    z_tensor,  # Rrs observation matrix
    num_clusters=2,  # number of classes
    epochs=10000,
    lr=1e-3,
    tau=1e-2,
    weight_norm_reg=1e-2,  # recommended to tune via CV
    max_tau=0.5,
    tau_growth=1.5,
):

    z_numpy = z_tensor.detach().cpu().numpy()
    scaler_z = StandardScaler().fit(z_numpy)
    z_std = scaler_z.transform(z_numpy)
    z_std_tensor = torch.tensor(z_std, dtype=torch.float32)

    input_bands = z_std_tensor.shape[1]
    num_samples = x_tensor.shape[1]

    x_numpy = x_tensor.detach().cpu().numpy()
    y_numpy = y_tensor.detach().cpu().numpy()

    qr_low = QuantileRegressor(quantile=0.01, fit_intercept=False, alpha=0.0)
    qr_high = QuantileRegressor(quantile=0.99, fit_intercept=False, alpha=0.0)
    qr_low.fit(x_numpy, y_numpy)
    qr_high.fit(x_numpy, y_numpy)
    b_low = qr_low.coef_[0]
    b_high = qr_high.coef_[0]

    beta_ols = LinearRegression(fit_intercept=False).fit(x_numpy, y_numpy).coef_[0]

    current_tau = float(tau)

    while True:
        model = ProjectionNet(input_dim=input_bands)
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        centers = nn.Parameter(torch.randn(num_clusters, 2) * 1e-1)
        betas = nn.Parameter(torch.full((num_clusters, num_samples), 1e-1))

        optimizer = torch.optim.Adam(
            [
                {"params": model.parameters(), "weight_decay": weight_norm_reg},
                {"params": [centers], "weight_decay": 0.0},
                {"params": [betas], "weight_decay": 0.0},
            ],
            lr=lr,
        )

        nan_flag = False

        for epoch in range(epochs):
            model.train()

            latent_feature = model(z_std_tensor)  # projection
            dist_sq = ((latent_feature.unsqueeze(1) - centers.unsqueeze(0)) ** 2).sum(dim=2)
            u = torch.softmax(-dist_sq / current_tau, dim=1)

            preds_k = x_tensor.matmul(betas.t())
            y_hat = (u * preds_k).sum(dim=1)

            loss = ((y_tensor - y_hat) ** 2).sum()

            if torch.isnan(loss) or torch.isnan(latent_feature).any() or torch.isnan(u).any():
                nan_flag = True
                break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # clamp betas to quantile bounds
            with torch.no_grad():
                betas.clamp_(min=b_low, max=b_high)

            # if epoch % 2000 == 0:
            #     print(f"epoch={epoch:4f}, loss={loss.item():.4f}")

        # avoid getting NaN results
        if nan_flag:
            current_tau *= tau_growth
            if current_tau > max_tau:
                current_tau /= tau_growth
                break
            continue

        break

    return {
        "model": model,
        "centers": centers.detach().cpu().numpy(),
        "betas": betas.detach().cpu().numpy(),
        "scaler_z": scaler_z,
        "current_tau": current_tau,
        "b_low": b_low,
        "b_high": b_high,
        "beta_ols": beta_ols,
    }



demo_data = joblib.load("soft_reconstruction_demo.joblib")

bbp_aop = demo_data["bbp_aop"]
aph_aop = demo_data["aph_aop"]
adg_aop = demo_data["adg_aop"]
wavelength = demo_data["wavelength"]


para_list = ["bbp", "aph", "adg"]

num_clusters = 2
epochs = 10000
lr = 1e-3
tau = 1e-2
weight_norm_reg = 1e-2

summary = {}
for para in para_list:
    df = demo_data[f"{para}_aop"].copy()
    z_numpy = df.filter(like="Rrse").values
    z_tensor = torch.tensor(z_numpy, dtype=torch.float32)

    # predictor: IOP(443)
    x_name = f"{para}443"
    x_numpy = df[x_name].values.reshape(-1, 1)
    x_tensor = torch.tensor(x_numpy, dtype=torch.float32)

    summary[para] = {}

    for band in wavelength:
        print(f"para={para}, band={band} ")
        if band == 443:
            tau_used = tau
            u = np.full((len(df), num_clusters), 1.0 / num_clusters, dtype=float)
            summary[para][band] = {
                "betas": betas,
                "centers": centers,
                "u": u,
                "tau": tau_used,
            }
            continue

        y_name = f"{para}{band}"
        y_numpy = df[y_name].values
        y_tensor = torch.tensor(y_numpy, dtype=torch.float32)

        result = soft_clustering_regression(x_tensor, y_tensor, z_tensor,
            num_clusters=num_clusters, epochs=epochs, lr=lr, tau=tau, weight_norm_reg=weight_norm_reg)

        betas = result["betas"]
        centers = result["centers"]
        scaler_z = result["scaler_z"]
        model = result["model"]
        tau_used = result["current_tau"]



        # compute soft memberships on the demo subset (for illustration)
        with torch.no_grad():
            z_std = scaler_z.transform(z_numpy)
            z_std_tensor = torch.tensor(z_std, dtype=torch.float32)
            q = model(z_std_tensor)
            centers_t = torch.tensor(centers, dtype=torch.float32)
            dist_sq = ((q.unsqueeze(1) - centers_t.unsqueeze(0)) ** 2).sum(dim=2)
            u = torch.softmax(-dist_sq / tau_used, dim=1).cpu().numpy()
            summary[para][band] = {
                "betas": betas,
                "centers": centers,
                "u": u,
                "tau": tau_used,
            }


print("\n All key parameters of the soft reconstruction have been stored in the variable of summary.")


