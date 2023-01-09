'''
    DS 5230
    Summer 2022
    HW5B_Problem_1_Simple_Sampling

    Apply rand() call to generate pseudo-uniform in [0,1] to sample
    from different distributions

    Hongyan Yang
'''


import numpy as np
import matplotlib.pyplot as plt
import collections

from scipy import stats

def binary_search(fun, target, low = -10, high = 10, tol = 1e-4):
    '''
    Provide solution to a monotonically increasing function
    '''
    if fun(low) > target or fun(high) < target:
        return None
    else:
        mid = (low + high) / 2
        while abs(fun(mid) - target) > tol:
            if fun(mid) > target:
                high = mid
            else:
                low = mid
            mid = (low + high) / 2
        return mid

def rejection_sampler(min, max, mu, sigma, N = 1000, gaussian = False):
    '''
    Implement rejection sampling from continuous distributions
    '''
    if gaussian:
        X = np.random.rand(N) * (max - min) * sigma + min * sigma + mu
        C = stats.norm(mu, sigma).pdf(mu)
        U = np.random.rand(N) * C
        fx = stats.norm.pdf(X, loc = mu, scale = sigma)
        X_sample = X[U < fx]
    else:
        X = np.random.rand(N) * (max - min) + min
        X_sample = X
    return X_sample

def rejection_sampler_2d(min, max, mu, sigma, N = 1000):
    '''
    Implement rejection sampling from 2-dim Gaussian Distribution
    '''
    sigma_X = np.sqrt(sigma[0][0])
    sigma_Y = np.sqrt(sigma[1][1])
    X = np.random.rand(N, 2)
    X[:,0] = X[:,0] * (max - min) * sigma_X + min * sigma_X
    X[:,1] = X[:,1] * (max - min) * sigma_Y + min * sigma_Y
    X = X + mu
    C = stats.multivariate_normal.pdf(mu, mean=mu, cov=sigma)
    U = np.random.rand(N) * C
    fx = stats.multivariate_normal.pdf(X, mean=mu, cov=sigma)
    X_sample = X[U < fx]
    return X_sample

def inverse_transform_sampler(min, max, mu, sigma, N = 1000, gaussian = False):
    '''
    Implement inverse sampling from continuous distributions
    '''
    X = np.random.rand(N)
    if gaussian:
        X_sample = list(map(lambda x:
                            binary_search(stats.norm(mu, sigma).cdf, x,
                                          low = mu + min * sigma,
                                          high = mu + max * sigma), X))
        X_sample = list(filter(None, X_sample))              
    else:
        f = lambda x: min + (max - min) * x
        X_sample = list(map(f, X))
    return X_sample

def inverse_transform_sampler_2d(min, max, mu, sigma, N = 1000):
    '''
    Implement inverse sampling from 2-dim Gaussian Distribution
    '''
    # Pick x value based on marginal distribution of x
    mu_X, sigma_X = mu[0], np.sqrt(sigma[0][0])
    X = np.random.rand(N)
    X_sample = list(map(lambda x:
                        binary_search(stats.norm(mu_X, sigma_X).cdf, x,
                                      low = mu_X + min * sigma_X,
                                      high = mu_X + max * sigma_X), X))
    X_sample = np.array(list(filter(None, X_sample)))
    # Pick y value based on conditional distribution of y|x
    mu_Y, sigma_Y = mu[1], np.sqrt(sigma[1][1])
    p = sigma[0][1] / (sigma_X * sigma_Y)
    mu_y = (X_sample - mu_X) * p * sigma_Y / sigma_X + mu_Y
    sigma_y = np.sqrt((1 - p ** 2) * (sigma_Y ** 2))
    Y = np.random.rand(len(X_sample))
    Y_sample = list()
    for i in range(len(X_sample)):
        y = binary_search(stats.multivariate_normal(mu_y[i], sigma_y).cdf,
                          Y[i], low = mu_y[i] + min * sigma_y,
                          high = mu_y[i] + max * sigma_y)
        Y_sample.append(y)
    # Construct sample points
    X_Y_sample = list(zip(X_sample, Y_sample))
    X_Y_sample = np.array(list(filter(lambda x: x[1] is not None, X_Y_sample)))
    return X_Y_sample

def generate_X(N = 300):
    '''
    Generate samples to form a discrete non-uniform distribution
    '''
    sample_list = np.array(range(1, N + 1))
    freq_list = np.array(range(N, 0, -1)) / np.sum(range(N, 0, -1))
    return sample_list, freq_list

def cdf_to_x(y, cdf):
    '''
    Implement inverse transform sampling to map cdf to x
    '''
    for i in range(len(cdf)):
        if y <= cdf[i]:
            return i

def inverse_transform_discrete(cdf, n = 20):
    '''
    Implement inverse sampling from a discrete non-uniform distribution
    '''
    X = np.random.rand(n)
    group_list = list(map(lambda x: cdf_to_x(x, cdf), X))
    freq_dict = dict(collections.Counter(group_list))
    return freq_dict

def plot_discrete_sample(sample, sample_pdf):
    '''
    Bar plot the sample with true pdf value
    '''
    plt.style.use('ggplot')
    fig, ax = plt.subplots()
    ax.bar(sample, sample_pdf, width=0.6, label='True pdf value',
           color ='#4682B4')
    ax.set_title('w/o_replacement sampling with Stevens method')
    ax.set_ylabel('pdf')
    ax.legend()
    plt.show()

def wo_replacement_sampler(N = 300, n = 20):
    '''
    Implement Stevens method to w/o_replacement sampling from a discrete
    non-uniform distribution
    '''
    sample_list, freq_list = generate_X(N = N)
    group_num = int(np.floor(N / n))
    group_dict, group_freq = dict(), np.zeros(group_num)
    for i in range(group_num - 1):
        group_dict[i] = sample_list[n * i:n + n * i]
        group_freq[i] = np.sum(freq_list[n * i:n + n * i])
    group_dict[group_num - 1] = sample_list[n * (group_num - 1):]
    group_freq[group_num - 1] = np.sum(freq_list[n * (group_num - 1):])
    group_cdf = [0] * len(group_freq)
    group_cdf[0] = group_freq[0]
    for i in range(1, len(group_freq)):
        group_cdf[i] = group_cdf[i - 1] + group_freq[i]
    # Get pick frequency for picked groups
    freq_dict = inverse_transform_discrete(group_cdf, n = n)
    sample = list()
    for key in freq_dict:
        members, m_size = group_dict[key], freq_dict[key]
        picked_members = list(np.random.choice(members, size=m_size,
                                               replace=False))
        sample.extend(picked_members)
    sample = sorted(sample)
    # Plot the sample with true pdf value
    sample_pdf = list(map(lambda x: freq_list[x - 1], sample))
    plot_discrete_sample(sample, sample_pdf)
    return sample

def plot_pdf(X, mu, sigma, gaussian = False):
    '''
    Plot simple sampling's distribution and pdf
    '''
    plt.style.use("ggplot")
    fig, ax0 = plt.subplots(ncols = 1, nrows = 1)
    values, bins, _ = ax0.hist(X, bins = 30, density = True, color = "#4682B4",
                               label = "Histogram of samples")
    if gaussian:
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        pdf_values = stats.norm.pdf(x = bin_centers, loc = mu, scale = sigma)
        ax0.plot(bin_centers, pdf_values, label = "PDF", color = "black")
    ax0.legend()
    ax0.set_title("PDF of samples from simple sampling")
    plt.show()

def plot_pdf_2d(X, mu, sigma):
    '''
    Plot simple sampling's 2-dim Gaussian Distribution
    '''
    plt.style.use("ggplot")
    fig, ax0 = plt.subplots(ncols = 1, nrows = 1)
    ax0.scatter(X[:,0], X[:,1], color = "#4682B4", alpha = 0.5,
                label = "Scatter plot of samples")
    # Contour plot of 2-dim gaussian with given parameters
    N = 200
    mu_X, sigma_X = mu[0], np.sqrt(sigma[0][0])
    mu_Y, sigma_Y = mu[1], np.sqrt(sigma[1][1])
    X = np.linspace(mu_X - 6 * sigma_X, mu_X + 6 * sigma_X, N)
    Y = np.linspace(mu_Y - 6 * sigma_Y, mu_Y + 6 * sigma_Y, N)
    X, Y = np.meshgrid(X, Y)
    pos = np.dstack((X, Y))
    rv = stats.multivariate_normal(mu, sigma)
    Z = rv.pdf(pos)
    ax0.contour(X, Y, Z, levels = 5)
    ax0.legend()
    ax0.set_title("PDF of 2-dim Gaussian Distribution from simple sampling")
    plt.show()

def main():
    print("# A) Sampling from uniform and gaussian distributions:\n")
    min, max, mu, sigma = -6, 6, -5, 2
    print("## Rejection sampling uniform distribution:")
    print(f"min = {min}, max = {max}\n")
    X = rejection_sampler(min, max, mu, sigma, N = 5000, gaussian = False)
    plot_pdf(X, mu, sigma, gaussian = False)
    print("## Rejection sampling gaussian distribution:")
    print(f"mu = {mu}, sigma = {sigma}\n")
    X = rejection_sampler(min, max, mu, sigma, N = 5000, gaussian = True)
    plot_pdf(X, mu, sigma, gaussian = True)
    print("## Inverse transform sampling uniform distribution:")
    print(f"min = {min}, max = {max}\n")
    X = inverse_transform_sampler(min, max, mu, sigma, N = 5000,
                                  gaussian = False)
    plot_pdf(X, mu, sigma, gaussian = False)
    print("## Inverse transform sampling gaussian distribution:")
    print(f"mu = {mu}, sigma = {sigma}\n")
    X = inverse_transform_sampler(min, max, mu, sigma, N = 5000,
                                  gaussian = True)
    plot_pdf(X, mu, sigma, gaussian = True)
    print()
    print("# B) Sampling from a 2-dim gaussian distribution:")
    mu, sigma = np.array([2, -15]), np.array([[50, 50], [50, 100]])
    print(f"mu = {mu}")
    print(f"sigma = {sigma}\n")
    print("## Rejection sampling:\n")
    X = rejection_sampler_2d(-6, 6, mu, sigma, N = 5000)
    plot_pdf_2d(X, mu, sigma)
    print("## Inverse transform sampling:\n")
    X = inverse_transform_sampler_2d(-6, 6, mu, sigma, N = 1000)
    plot_pdf_2d(X, mu, sigma)
    print()
    print("# C) w/o replacement sampling from a discrete " \
          "non-uniform distribution:")
    N, n = 300, 20
    print(f"N = {N}, n = {n}\n")
    run = "Y"
    while run == "Y":
        wo_replacement_sampler(N = N, n = n)
        run = input("Would you like to resample? (Y/ N) ").upper()
    print("\n")
    print("Complete.")


if __name__ == "__main__":
    main()
