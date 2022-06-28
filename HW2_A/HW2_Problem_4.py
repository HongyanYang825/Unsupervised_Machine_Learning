'''
    DS 5230
    Summer 2022
    HW2_Problem_4_Gaussian_Mixture_on_real_data

    Train and test GMM for SPAMBASE datasets

    Hongyan Yang
'''


import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from HW2_Problem_2 import *
from HW2_Problem_3 import *
from HW2_constants import *

def get_aic(vectors, tol = 1e-4, max_iter = 10, cosine_distances = False):
    '''
    Calculate AIC given different number of clusters
    '''
    aic_dict = dict()
    for k in range(2, 21):
        iter, iter_list = 0, list()
        while iter < max_iter:
            memberships, centroids_diff = initial_assign(vectors, k - 1), 1
            centroids = m_step(vectors, memberships, k - 1)
            while centroids_diff > tol:
                memberships = e_step(vectors, centroids, cosine_dist = cosine_distances)
                new_centroids = m_step(vectors, memberships, k - 1)
                centroids_diff = abs(np.linalg.norm(new_centroids) - np.linalg.norm(centroids))
                centroids = new_centroids
            try:
                obj_mat = np.stack([np.sum(euclidean_distances(vectors[np.where(memberships == i)],
                                                               new_centroids[i].reshape(1, -1)) ** 2)
                                    for i in range(k)], axis = 0)
                aic = np.sum(obj_mat) + 2 * new_centroids.shape[0] * new_centroids.shape[1]
                iter_list.append(aic)
                iter += 1
            except:
                continue
        aic_dict[k] = min(iter_list)
    return  aic_dict

def plot_aic(aic_dict):
    '''
    Plot AIC given different number of clusters
    '''
    plt.style.use('_mpl-gallery')
    x = np.array(list(aic_dict.keys()))
    y = np.array(list(aic_dict.values()))
    plt.plot(x, y, label = "AIC for chosen k")
    plt.xticks(np.arange(min(x), max(x) + 1, 1.0))
    plt.legend(loc = 'upper right')
    plt.xlabel('number of clusters')
    plt.ylabel('AIC')
    plt.title('AIC for different number of clusters')
    plt.tight_layout()
    plt.show()

def predict_labels(vectors_pca, k_1, k_0, prior_1, prior_0, mu_1, sigma_1, mix_prob_1,
                   mu_0, sigma_0, mix_prob_0):
    '''
    Apply GMM to predict instances' labels
    '''
    densities_1 = np.stack([multivariate_normal.pdf(vectors_pca, mu_1[i], sigma_1[i])
                            for i in range(k_1)], axis = 1)
    p_x_spam = np.matmul(densities_1, mix_prob_1) * prior_1
    densities_0 = np.stack([multivariate_normal.pdf(vectors_pca, mu_0[i], sigma_0[i])
                            for i in range(k_0)], axis = 1)
    p_x_nonspam = np.matmul(densities_0, mix_prob_0) * prior_0
    predicted_labels = p_x_spam / p_x_nonspam
    predicted_labels[predicted_labels > 1] = 1
    predicted_labels[predicted_labels < 1] = 0
    return predicted_labels

def main():
    print("### Spambase Datasets")
    spambase_data = np.loadtxt("spambase.data", delimiter=",", dtype = float)
    spambase_labels = spambase_data[:, -1]
    prior_1, prior_0 = (sum(spambase_labels == 1)/ len(spambase_labels),
                        sum(spambase_labels == 0)/ len(spambase_labels))
    spambase_data = np.delete(spambase_data, -1, 1)
    spam_email = spambase_data[np.where(spambase_labels == 1)]
    nonspam_email = spambase_data[np.where(spambase_labels == 0)]
    print()
    print("## Determining the optimal number of clusters...")
    plot_aic(get_aic(spam_email))
    plot_aic(get_aic(nonspam_email))
    print("Optimal number of clusters for spam_email sub_datasets is 5")
    print("Optimal number of clusters for nonspam_email sub_datasets is 6")
    print()
    print("## Applying PCA to spambase datasets for dimensionality reduction...")
    pipe = Pipeline([("scaler", Normalizer()), ("pca", PCA(n_components = 5))])
    spambase_pca = pipe.fit_transform(spambase_data)
    print("Explained variance ratio of spambase datasets:")
    print(pipe[1].explained_variance_ratio_)
    print("Number of principal components of spambase datasets is 3")
    print()
    print("## Fitting GMM to both preprossed sub_datasets...")
    pipe = Pipeline([("scaler", Normalizer()), ("pca", PCA(n_components = 6))])
    spambase_pca = pipe.fit_transform(spambase_data)
    spam_email_pca = spambase_pca[np.where(spambase_labels == 1)]
    mu_1, sigma_1, mix_prob_1, max_obj_1 = gmm_fit(spam_email_pca, 5, tol = 1e-4, max_iter = 300)
    #mu_1, sigma_1, mix_prob_1 = MU_1, SIGMA_1, MIX_PROB_1
    nonspam_email_pca = spambase_pca[np.where(spambase_labels == 0)]
    mu_0, sigma_0, mix_prob_0, max_obj_0 = gmm_fit(nonspam_email_pca, 6, tol = 1e-4, max_iter = 300)
    #mu_0, sigma_0, mix_prob_0 = MU_0, SIGMA_0, MIX_PROB_0    
    predicted_labels = predict_labels(spambase_pca, 5, 6, prior_1, prior_0, mu_1, sigma_1, mix_prob_1,
                                      mu_0, sigma_0, mix_prob_0)[:, 0]
    accuracy = round(sum(predicted_labels == spambase_labels) / len(spambase_labels), 4)
    precision = round(sum(spambase_labels[np.where(predicted_labels == 1)] == 1)
                      / sum(predicted_labels == 1), 4)
    recall = round(sum(spambase_labels[np.where(predicted_labels == 1)] == 1)
                      / sum(spambase_labels == 1), 4)
    print("Spambase Datasets testing performance:")
    print(f"GMM fitting's Accuracy: {accuracy}")
    print(f"GMM fitting's Precision: {precision}")
    print(f"GMM fitting's Recall: {recall}")


if __name__ == "__main__":
    main()
