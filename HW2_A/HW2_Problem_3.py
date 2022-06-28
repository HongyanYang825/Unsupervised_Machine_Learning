'''
    DS 5230
    Summer 2022
    HW2_Problem_3_Gaussian_Mixture_on_toy_data

    Train and test GMM for simulated datasets

    Hongyan Yang
'''


import numpy as np

from scipy.stats import multivariate_normal
from HW2_Problem_2 import *

def k_means_initial_assign(vectors, k, tol = 1e-4, max_iter = 10, cosine_distances = False):
    '''
    Apply K_Means to initialize GMM parameters efficiently
    '''
    iter, iter_dict = 0, dict()
    sub_vectors = vectors[np.random.choice(len(vectors), 100 * k, replace = False)]
    while iter < max_iter:
        memberships, centroids_diff = initial_assign(sub_vectors, k - 1), 1
        centroids = m_step(sub_vectors, memberships, k - 1)
        while centroids_diff > tol:
            memberships = e_step(sub_vectors, centroids, cosine_dist = cosine_distances)
            new_centroids = m_step(sub_vectors, memberships, k - 1)
            centroids_diff = abs(np.linalg.norm(new_centroids) - np.linalg.norm(centroids))
            centroids = new_centroids
        try:
            init_mix_prob = np.stack([len(memberships[np.where(memberships == i)])
                                      / len(sub_vectors)
                                      for i in range(k)], axis = 0).reshape(-1, 1)
            init_mu = new_centroids
            init_sigma = np.stack([np.cov(sub_vectors[np.where(memberships == i)].T)
                                   for i in range(k)], axis = 0)
            obj_mat = np.stack([np.sum(euclidean_distances(sub_vectors[np.where(memberships == i)],
                                                           init_mu[i].reshape(1, -1)) ** 2)
                                for i in range(k)], axis = 0)
            objective = np.sum(obj_mat)
            iter_dict[objective] = init_mix_prob, init_mu, init_sigma
            iter += 1
        except:
            continue
    min_objective = min(iter_dict.keys())
    parameters = iter_dict[min_objective]
    return parameters[0], parameters[1], parameters[2]

def objective_funcition(vectors, k, mu, sigma, mix_prob):
    '''
    Objective function of the GMM
    '''
    densities = np.stack([multivariate_normal.pdf(vectors, mu[i], sigma[i])
                          for i in range(k)], axis = 1)
    objective = np.sum(np.log(np.matmul(densities, mix_prob)))
    return objective

def gmm_e_step(vectors, k, mu, sigma, mix_prob):
    '''
    E step of the GMM
    '''
    densities = np.stack([multivariate_normal.pdf(vectors, mu[i], sigma[i])
                          for i in range(k)], axis = 1)
    row_sum = np.sum(densities, axis = 1)
    membership_prob = densities / row_sum[:, None]
    return membership_prob

def gmm_m_step(vectors, k, mem_prob, dim = 1):
    '''
    M step of the GMM
    '''
    mu = (np.matmul(vectors.T, mem_prob) / np.sum(mem_prob, axis = 0)).T
    sigma = np.stack([(np.matmul((vectors - mu[i]).T, ((vectors - mu[i]) * mem_prob[:, i, None]))
                       / (dim * np.sum(mem_prob[:, i]))) for i in range(k)], axis = 0)
    mix_prob = np.average(mem_prob, axis = 0)[:, np.newaxis]
    return mu, sigma, mix_prob

def gmm_fit(vectors, k, tol = 1e-4, max_iter = 300):
    '''
    Fit GMM to datasets and return the best performed parameters
    '''
    cov_adj = np.stack([1e-6 * np.identity(vectors.shape[1]) for i in range(k)])
    iter, mem_prob_diff, iter_dict = 0, 1, dict()
    while iter < max_iter:
        try:
            mix_prob, mu, sigma = k_means_initial_assign(vectors, k)
            sigma += cov_adj
            mem_prob = gmm_e_step(vectors, k, mu, sigma, mix_prob)
            while mem_prob_diff > tol:
                mu, sigma, mix_prob = gmm_m_step(vectors, k, mem_prob, dim = 1)
                sigma += cov_adj
                new_mem_prob = gmm_e_step(vectors, k, mu, sigma, mix_prob)
                mem_prob_diff = abs(np.linalg.norm(new_mem_prob) - np.linalg.norm(mem_prob))
                mem_prob = new_mem_prob
            objective = objective_funcition(vectors, k, mu, sigma, mix_prob)
            iter_dict[objective] = mu, sigma, mix_prob
            iter += 1
        except:
            continue
    max_obj = max(iter_dict.keys())
    return iter_dict[max_obj][0], iter_dict[max_obj][1], iter_dict[max_obj][2], max_obj

def main():
    print("### run the EM algorithm to recover parameters of TWO mixtures\n")
    vectors_2gm = np.loadtxt("2gaussian.txt", dtype = float)
    mu_2, sigma_2, mix_prob_2, max_obj_2 = gmm_fit(vectors_2gm, 2, tol = 1e-4, max_iter = 3)
    print("## mean:")
    print(mu_2)
    print()
    print("## covariance:")
    print(sigma_2)
    print()
    print("## mix probabilities:")
    print(mix_prob_2)
    print("\n")
    print("### run the EM algorithm to recover parameters of THREE mixtures\n")
    vectors_3gm = np.loadtxt("3gaussian.txt", dtype = float)
    mu_3, sigma_3, mix_prob_3, max_obj_3 = gmm_fit(vectors_3gm, 3, tol = 1e-4, max_iter = 3000)
    print("## mean:")
    print(mu_3)
    print()
    print("## covariance:")
    print(sigma_3)
    print()
    print("## mix probabilities:")
    print(mix_prob_3)


if __name__ == "__main__":
    main()
