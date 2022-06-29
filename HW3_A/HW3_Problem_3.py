'''
    DS 5230
    Summer 2022
    HW3A_Problem_3_Implement_PCA_on_MNIST

    Run own PCA implementation to train and test L2-reg Logistic Regression
    and Decision Trees on MNIST datasets

    Hongyan Yang
'''


from HW3_Problem_1 import *
from numpy.linalg import eigh

def pca_fit_transform(vectors, n_components):
    '''
    Fit the model with vectors and apply the dimensionality reduction
    '''
    vectors_copy = np.copy(vectors)
    mu = np.mean(vectors_copy, axis = 0)
    vectors_copy = vectors_copy - mu
    cov_matrix = np.matmul(vectors_copy.T, vectors_copy)
    eigen_values, eigen_vectors = eigh(cov_matrix)
    eigen_values = np.flip(eigen_values)
    eigen_vectors = np.flip(eigen_vectors, axis = 1)
    explained_variance_ratio = eigen_values[:n_components] / np.sum(eigen_values)                      
    components = np.matmul(vectors_copy, eigen_vectors[:, :n_components])
    return components, explained_variance_ratio, eigen_values, eigen_vectors

def pca_transform(vectors, n_components, eigen_values, eigen_vectors):
    '''
    Apply dimensionality reduction to vectors

    Vectors is projected on the first principal components previously trained
    '''
    vectors_copy = np.copy(vectors)
    mu = np.mean(vectors_copy, axis = 0)
    vectors_copy = vectors_copy - mu
    explained_variance_ratio = eigen_values[:n_components] / np.sum(eigen_values)                      
    components = np.matmul(vectors_copy, eigen_vectors[:, :n_components])
    return components, explained_variance_ratio

def main():
    print("### Load and normalize the MNIST Dataset\n")
    train_f, train_labels, test_f, test_labels = parse_MNIST(CWD + "/mnist.npz")
    train_f, test_f = edit_normalize_MNIST(train_f), edit_normalize_MNIST(test_f)
    print("## Run a PCA-library to get data on D = 5 features:\n")
    results = pca_fit_transform(train_f, 5)
    train = results[0]
    test = pca_transform(test_f, 5, results[2], results[3])[0]
    print("# Train and test L2-reg Logistic Regression and analyze features:\n")
    logistic_reg_clf(train, train_labels, test, test_labels, tol = 1e0, max_iter = 800)
    print()
    print("# Train and test Decision Trees and analyze features:\n")
    decision_tree_clf(train, train_labels, test, test_labels)
    print()
    print("## Run a PCA-library to get data on D = 20 features:\n")
    results = pca_fit_transform(train_f, 20)
    train = results[0]
    test = pca_transform(test_f, 20, results[2], results[3])[0]
    print("# Train and test L2-reg Logistic Regression and analyze features:\n")
    logistic_reg_clf(train, train_labels, test, test_labels, tol = 1e0, max_iter = 800)
    print()
    print("# Train and test Decision Trees and analyze features:\n")
    decision_tree_clf(train, train_labels, test, test_labels)


if __name__ == "__main__":
    main()
