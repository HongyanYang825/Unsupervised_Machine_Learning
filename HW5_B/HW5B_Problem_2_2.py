'''
    DS 5230
    Summer 2022
    HW5B_Problem_2_2_Conditional_Sampling

    Implement Gibbs Sampling for a multidim gaussian generative joint

    Hongyan Yang
'''


from HW5B_Problem_1 import *

def x_given_y(y, mu_X, mu_Y, sigma_X, sigma_Y, rho):
    '''
    Sample x given y
    '''
    mu_x = mu_X + (y - mu_Y) * rho * sigma_X / sigma_Y
    sigma_x = np.sqrt((1 - rho ** 2) * sigma_X ** 2)
    return np.random.normal(mu_x, sigma_x)

def y_given_x(x, mu_X, mu_Y, sigma_X, sigma_Y, rho):
    '''
    Sample y given x
    '''
    mu_y = mu_Y + (x - mu_X) * rho * sigma_Y / sigma_X
    sigma_y = np.sqrt((1 - rho ** 2) * sigma_Y ** 2)
    return np.random.normal(mu_y, sigma_y)

def gibbs_sampler(mu, sigma, N = 1000):
    '''
    Implement Gibbs Sampling for a multidim gaussian generative joint
    '''
    mu_X, sigma_X = mu[0], np.sqrt(sigma[0][0])
    mu_Y, sigma_Y = mu[1], np.sqrt(sigma[1][1])
    rho = sigma[0][1] / (sigma_X * sigma_Y)
    x_list, y_list = [None] * N, [None] * N
    # Pick x value based on marginal distribution of x to initialize
    x_list[0] = np.random.normal(mu_X, sigma_X)
    y_list[0] = y_given_x(x_list[0], mu_X, mu_Y, sigma_X, sigma_Y, rho)
    for i in range(1, N):
        x_list[i] = x_given_y(y_list[i - 1], mu_X, mu_Y, sigma_X, sigma_Y, rho)
        y_list[i] = y_given_x(x_list[i], mu_X, mu_Y, sigma_X, sigma_Y, rho)
    # Construct sample points
    X_Y_sample = np.array(list(zip(x_list, y_list)))
    return X_Y_sample

def main():
    mu = np.array([2, -15])
    sigma = np.array([[50, 50], [50, 100]])
    print(f"mu = {mu}")
    print(f"sigma = {sigma}\n")
    X = gibbs_sampler(mu, sigma, N = 1000)
    plot_pdf_2d(X, mu, sigma)


if __name__ == "__main__":
    main()
