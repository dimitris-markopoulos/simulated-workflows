from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA

#======= 1. GENERATE DATA =======

def data_generator (seed, n, dim = 9):
    mu1 = [1, 1, 1, 0, 0, 0, 0, 0, 0]
    mu2 = [0, 0, 0, 0, 0, 0, 1, 1, 1]
    mu3 = [0, 0, 0, 1, 1, 1, 0, 0, 0]
    sigma1 = np.diag([1, 1, 1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    sigma2 = np.diag([0.1 , 0.1, 0.1, 0.1, 0.1, 0.1, 1, 1, 1])
    sigma3 = np.diag([0.1, 0.1, 0.1, 1, 1, 1, 0.1, 0.1, 0.1])
    np.random.seed(seed)
    rand_int = np.random.choice([1 , 2 , 3], size = n)
    unique_values, counts = np.unique(rand_int, return_counts = True)
    datapoints = np.zeros((0, dim))
    labels = np.array([])
    for i, (uv, mu, sigma) in enumerate(zip(unique_values, [mu1, mu2, mu3], [sigma1, sigma2, sigma3])):
        datapoints = np.vstack((datapoints, np.random.multivariate_normal(mu, sigma, size = counts[i])))
        labels = np.hstack((labels, uv*np.ones(counts[i])))
    shuff = np.random.permutation(len(labels))
    return datapoints[shuff], labels[shuff]

#======= 2. DEFINE UTILS =======

def fit_score_kmeans(
        K : int, # hyperparameter
        X : pd.DataFrame, 
        y : pd.DataFrame,
    ) -> float:
    """
    Returns adjusted randscore.
    """
    
    # same KMeans initialization for the paired runs so differences are due to PCA, not luck in centroid init
    model = KMeans(n_clusters=K, n_init=50, random_state=0) 
    fit   = model.fit(X)

    y_pred = fit.predict(X)
    score = adjusted_rand_score(
        labels_true = y.squeeze(),
        labels_pred = y_pred.squeeze()
    )
    return score

#======= 3. ITERATION + FIT =======

def iterative_fit_score_kmeans(
    K : int = 3, # hyperparameter
    n_vals : List[int] = [100, 200, 300, 400, 500],
    seed_vals : List[int] = [x for x in range(1,101)],
    verbose : bool = True,
    perform_PCA : bool = False
):
    """
    Iteratively fit and score each length of data generated and across multiple seeds.
    Returns two dictionaries containing the mean rand score across each fold and the iterative scores.
    """
    score_dict = {}
    cv_dict = {n:[] for n in n_vals}
    for n in n_vals:
        for seed in seed_vals:
            raw_data = data_generator(seed=seed, n=n, dim=9)
            X_df = pd.DataFrame(
                raw_data[0], 
                columns=[f'feature_{x}' for x in range(raw_data[0].shape[1])]
            )
            y_df = pd.DataFrame(
                raw_data[1], 
                columns=['labels']
            )
            if perform_PCA == True:
                pca = PCA(n_components=K)
                X_pca_df = pd.DataFrame(pca.fit_transform(X_df))
                params_dict = {
                    'K' : K,
                    'X' : X_pca_df,
                    'y' : y_df,
                }
            
            else: # NO PCA
                params_dict = {
                    'K' : K,
                    'X' : X_df,
                    'y' : y_df,
                }
            iter_score = fit_score_kmeans(**params_dict)
            cv_dict[n].append(iter_score)
        
        score_dict[n] = np.array(cv_dict[n]).mean()

        if verbose:
            print(f'*** n{n} : chunk : size = {len(seed_vals)} : complete ***')

    return (score_dict, cv_dict)

def summarize(cv_dict):
    ns = sorted(cv_dict.keys())
    means = [np.mean(cv_dict[n]) for n in ns]
    stds  = [np.std(cv_dict[n], ddof=1) for n in ns]  # std, not SE
    return ns, means, stds

#======= 4. VIZ =======

def viz_rand_score_over_folds(
    figure, axes,
    K : int = 3, # hyperparameter
    n_vals : List[int] = [100, 200, 300, 400, 500],
    seed_vals : List[int] = [x for x in range(1,101)],
    verbose : bool = True,
    perform_PCA : bool = False,
    label : str = "",
    color : str = "red"
):
    """
    Wrap iterative_fit_score_kmeans and plot scores across folds.
    Returns fig and axes
    """
    ## WRAP
    score_dict, cv_dict = (
        iterative_fit_score_kmeans(
            K = K,
            n_vals = n_vals,
            seed_vals = seed_vals,
            verbose = verbose,
            perform_PCA = perform_PCA
        )
    )

    ## PLOT RESULTS
    m = len(cv_dict.keys())
    for ax, n_val in zip(axes,cv_dict.keys()):
        scores_array = np.array(cv_dict[n_val])
        mean_ = scores_array.mean()
        std_ = scores_array.std()
        ax.plot(scores_array, color=color)
        ax.set_title(f"N = {n_val}", fontsize=12, pad=10)

        ## mean of folds
        ax.axhline(mean_, color='red',label=f'avg_rand: {mean_:.2f}')

        ## 95 % error CI band
        thres = std_ * (1.96/np.sqrt(len(scores_array)))
        lower_bound, upper_bound = mean_ - thres, mean_ + thres
        ax.axhline(lower_bound, color='red', linestyle = ':',label=f'lb: {lower_bound:.2f}')
        ax.axhline(upper_bound, color='red', linestyle = ':',label=f'ub: {upper_bound:.2f}')
        
        ax.legend(loc='lower right')

        ax.fill_between(
        range(len(scores_array)),
        lower_bound, upper_bound,
        color='red', alpha=0.1, label='95% CI'
        )

    if label:
        figure.suptitle(label, fontsize=16, fontweight="bold", y=1.0)

    figure.supxlabel("iterations", fontsize=14, fontweight="bold")
    figure.supylabel("adjusted_rand_score", fontsize=14, fontweight="bold")
    figure.tight_layout()
    figure.subplots_adjust(left=0.05)

    return figure, axes


def viz_rand_score_over_folds_superimposed(
    K : int = 3,
    n_vals : List[int] = [100, 200, 300, 400, 500],
    seed_vals : List[int] = [x for x in range(1,101)],
    verbose : bool = True
):
    """
    Superimposed visual + could not wrap above function so had to create another to superimpose.
    """

    # Run once without PCA
    _, cv_raw = iterative_fit_score_kmeans(
        K=K, n_vals=n_vals, seed_vals=seed_vals,
        verbose=verbose, perform_PCA=False
    )
    # Run once with PCA
    _, cv_pca = iterative_fit_score_kmeans(
        K=K, n_vals=n_vals, seed_vals=seed_vals,
        verbose=verbose, perform_PCA=True
    )

    m = len(n_vals)
    fig, axes = plt.subplots(1, m, figsize=(5*m, 5), sharey=True)

    for j, n_val in enumerate(n_vals):
        ax = axes[j]

        # NO reduction raw curve
        scores_raw = np.array(cv_raw[n_val])
        ax.plot(scores_raw, color="tab:blue", alpha=0.6)

        # WITH PCA raw curve
        scores_pca = np.array(cv_pca[n_val])
        ax.plot(scores_pca, color="tab:orange", alpha=0.6)

        mean_raw = scores_raw.mean()
        mean_pca = scores_pca.mean()
        ax.axhline(mean_raw, color="tab:blue", linestyle="--", linewidth=2, label=f"~PCA mean: {mean_raw:.2f}")
        ax.axhline(mean_pca, color="tab:orange", linestyle="--", linewidth=2, label=f"PCA mean: {mean_pca:.2f}")

        ax.legend(fontsize=9, loc="lower right", frameon=False)
        ax.set_title(f"N = {n_val}", fontsize=12)
    fig.supxlabel("iterations", fontsize=14, fontweight="bold")
    fig.supylabel("adjusted_rand_score", fontsize=14, fontweight="bold")
    fig.suptitle("Clustering Performance: ~PCA vs PCA", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15, left=0.05)

    return fig, axes


if __name__ == '__main__':
    pass