### Author: Pongpisit Thanasutives ###

import os
from itertools import combinations
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from tqdm import trange
from scipy import io as sio
from scipy.stats import uniform, norm
import pysindy as ps
import pocomc as pc
from bayesian_model_evidence import log_evidence


def TopRsq(X_full, y, m, n_tops=25):
    n_feats = X_full.shape[-1]
    r_scores = []
    models = []
    for comb in combinations(range(n_feats), m):
        comb = list(comb)
        active_indices = np.zeros(n_feats)
        active_indices[comb] = 1
        X_sub = X_full[:, comb]
        lr = LinearRegression(fit_intercept=False).fit(X_sub, y)
        R2 = lr.score(X_sub, y)
        r_scores.append(R2)
        models.append(active_indices)
    r_scores = np.array(r_scores)
    r_argsort = np.argsort(r_scores)[::-1][:n_tops]
    r_scores = r_scores[r_argsort]
    models = np.array(models).T
    models = models[:, r_argsort]
    rating = np.dot(models, r_scores)
    return models, r_scores, rating


def comprehensive_search(X_full, y, max_support_size=8, n_tops=None, threshold=0.75):
    X = X_full.copy()
    n_feats = X_full.shape[-1]
    n_tops = int(np.ceil(n_feats / 2)) if n_tops is None else n_tops
    ratings = np.zeros((n_feats, max_support_size))
    search = True
    support_size = 1
    optimal_indices = None
    active_indices = [_ for _ in range(n_feats)]
    while search and support_size <= max_support_size:
        _, _, rating = TopRsq(X, y, m=support_size, n_tops=n_tops)
        rating = rating / rating.max()
        ratings[:, support_size - 1][active_indices] = rating
        if support_size >= 2:
            i0 = np.where(
                ratings[:, support_size - 1] + ratings[:, support_size - 2] == 0.0
            )[0]
            active_indices = [_ for _ in active_indices if _ not in set(i0)]
            X = X_full[:, active_indices]
            i1 = np.where(ratings[:, support_size - 1] > threshold)[0]
            i2 = np.where(ratings[:, support_size - 2] > threshold)[0]
            if len(i1) == len(i2) and np.all(i1 == i2):
                search = False
                optimal_indices = i1
        support_size += 1
    return optimal_indices, ratings[:, : support_size - 1]


n_experiments = 100
n_samples = 10000
n_features = 8
n_informative = 2

threshold = 0.25
max_support_size = 8

success = 0
for i in trange(n_experiments):
    X_train, y_train = make_regression(
        n_samples=n_samples, n_features=n_features, n_informative=n_informative
    )
    top_models, _, _ = TopRsq(X_train, y_train, m=n_informative)
    true_indices = np.where(top_models[:, 0] > 0)[0]
    est_indices, ratings = comprehensive_search(
        X_train, y_train, max_support_size=max_support_size, threshold=threshold
    )
    if (
        est_indices is not None
        and len(true_indices) == len(est_indices)
        and np.all(true_indices == est_indices)
    ):
        success += 1

success / n_experiments

# ### PDE ###

data_path = "./Datasets/"
data = sio.loadmat(os.path.join(data_path, "burgers.mat"))
u_clean = (data["usol"]).real
u = u_clean.copy()
x = (data["x"][0]).real
t = (data["t"][:, 0]).real
dt = t[1] - t[0]
dx = x[2] - x[1]

np.random.seed(0)
noise_type = "gaussian"
noise_lv = float(50)
print("Noise level:", noise_lv)
noise = 0.01 * np.abs(noise_lv) * (u.std()) * np.random.randn(u.shape[0], u.shape[1])
u = u + noise
u = np.load("./Denoised_data/burgers_gaussian50_bm3d.npy")

xt = np.array([x.reshape(-1, 1), t.reshape(1, -1)], dtype=object)
X, T = np.meshgrid(x, t)
XT = np.asarray([X, T]).T

function_library = ps.PolynomialLibrary(degree=2, include_bias=False)

weak_lib = ps.WeakPDELibrary(
    function_library=function_library,
    derivative_order=3,
    spatiotemporal_grid=XT,
    include_bias=True,
    diff_kwargs={"is_uniform": True},
    K=10000,
)

X_pre = np.array(weak_lib.fit_transform(np.expand_dims(u, -1)))
y_pre = weak_lib.convert_u_dot_integral(np.expand_dims(u, -1))
N = len(y_pre)

effective_indices, rating = comprehensive_search(
    X_pre, y_pre, max_support_size=max_support_size, threshold=0.75
)
effective_indices, rating

from exhaustive_selection import best_subset
from functools import partial
from p_tqdm import p_map

best_subset_results = p_map(partial(best_subset, X_pre, y_pre), [1, 2, 3, 4, 5])
best_subset_results

# ### Exat model evidence ###

bme = [
    log_evidence(X_pre, y_pre, effective_indices, v=1e-2, standardize=False)
    for effective_indices in best_subset_results
]
bme, np.argmax(bme) + 1

# ### Bayes factor ###

effective_indices = [4, 6]
X_pre_sub = X_pre[:, effective_indices].copy().T


def log_likelihood(param):
    global X_pre_sub, y_pre, N
    ssr = np.sum(np.abs(param @ X_pre_sub - y_pre.flatten()) ** 2, axis=-1)

    def ssr2llf(ssr, nobs):
        nobs2 = nobs / 2.0
        llf = -nobs2 * np.log(2 * np.pi) - nobs2 * np.log(ssr / nobs) - nobs2
        return llf

    return ssr2llf(ssr, N)


n_dim = len(effective_indices)
prior = pc.Prior(n_dim * [norm(0, 1)])
sampler = pc.Sampler(
    prior=prior,
    likelihood=log_likelihood,
    vectorize=True,
)
sampler.run()
simple_samples, simple_weights, _, _ = sampler.posterior()
logz_simple, logz_err_simple = sampler.evidence()

effective_indices = [4, 5, 6]
X_pre_sub = X_pre[:, effective_indices].copy().T


def log_likelihood(param):
    global X_pre_sub, y_pre, N
    ssr = np.sum(np.abs(param @ X_pre_sub - y_pre.flatten()) ** 2, axis=-1)

    def ssr2llf(ssr, nobs):
        nobs2 = nobs / 2.0
        llf = -nobs2 * np.log(2 * np.pi) - nobs2 * np.log(ssr / nobs) - nobs2
        return llf

    return ssr2llf(ssr, N)


n_dim = len(effective_indices)
prior = pc.Prior(n_dim * [norm(0, 1)])
sampler = pc.Sampler(
    prior=prior,
    likelihood=log_likelihood,
    vectorize=True,
)
sampler.run()
extended_samples, extended_weights, _, _ = sampler.posterior()
logz_extended, logz_err_extended = sampler.evidence()

# Bayes factor of extended to simple model
BF = np.exp(logz_extended - logz_simple)
if BF > 1:
    print("The extended model is more probable than the simple model.")
else:
    print("The simple model is more probable than the extended model.")

np.dot(simple_weights, simple_samples)

np.dot(extended_weights, extended_samples)
