'''
    DS 5230
    Summer 2022
    HW3A_Problem_2_PCA_library_on_MNIST

    Run a PCA library to train and test L2-reg Logistic Regression and
    Decision Trees on MNIST and Spambase datasets

    Hongyan Yang
'''


from HW3_Problem_1 import *
from sklearn.decomposition import PCA

def plot_pca_score(train_f, test_f, train_labels, test_labels, names):
    '''
    Plot scores given different number of PCA dimmesnsions
    '''
    x, y = list(), list()
    for i in range(3, 21):
        print(f"For number of PCA dimmesnsions = {i}:\n")
        pca = PCA(n_components = i)
        train = pca.fit_transform(train_f)
        test = pca.transform(test_f)
        score = logistic_reg_clf(train, train_labels, test, test_labels, tol = 1e0,
                                 max_iter = 1000, names = names, show_names = True)
        print()
        x.append(i)
        y.append(score)
    plt.style.use("_mpl-gallery")
    plt.plot(x, y, label = "Purity scores")
    plt.xticks(np.arange(min(x), max(x) + 1, 1.0))
    plt.legend(loc = "upper right")
    plt.xlabel("number of PCA dimmesnsions")
    plt.ylabel("Purity")
    plt.title("Purity given different number of PCA dimmesnsions")
    plt.tight_layout()
    plt.show()

def main():
    print("### A) Load and normalize the MNIST Dataset\n")
    train_f, train_labels, test_f, test_labels = parse_MNIST(CWD + "/mnist.npz")
    train_f, test_f = edit_normalize_MNIST(train_f), edit_normalize_MNIST(test_f)
    print("## Run a PCA-library to get data on D = 5 features:\n")
    pca = PCA(n_components = 5)
    train = pca.fit_transform(train_f)
    test = pca.transform(test_f)
    print("# Train and test L2-reg Logistic Regression and analyze features:\n")
    logistic_reg_clf(train, train_labels, test, test_labels, tol = 1e0, max_iter = 800)
    print()
    print("# Train and test Decision Trees and analyze features:\n")
    decision_tree_clf(train, train_labels, test, test_labels)
    print()
    print("## Run a PCA-library to get data on D = 20 features:\n")
    pca = PCA(n_components = 20)
    train = pca.fit_transform(train_f)
    test = pca.transform(test_f)
    print("# Train and test L2-reg Logistic Regression and analyze features:\n")
    logistic_reg_clf(train, train_labels, test, test_labels, tol = 1e0, max_iter = 800)
    print()
    print("# Train and test Decision Trees and analyze features:\n")
    decision_tree_clf(train, train_labels, test, test_labels)
    print("\n")
    print("### B) Load and normalize the Spambase Dataset\n")
    spambase_data = np.loadtxt("spambase.data", delimiter=",", dtype = float)
    labels = spambase_data[:, -1]
    spambase_data = np.delete(spambase_data, -1, 1)
    spambase_data = normalize_zero_mean_unit_variance(spambase_data)
    data = train_test_split(spambase_data, labels, test_size = 0.25, random_state = 0)
    train_f, test_f, train_labels, test_labels = data
    names = parse_spambase_names(CWD + "/spambase.names")
    plot_pca_score(train_f, test_f, train_labels, test_labels, names)


if __name__ == "__main__":
    main()
