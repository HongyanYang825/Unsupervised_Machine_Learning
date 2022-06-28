'''
    DS 5230
    Summer 2022
    HW1_Problem_4_Train_and_Test_KNN_Classification

    Train and test KNN classification for 20NG and MNIST datasets
    Report both training performance and testing performance

    Hongyan Yang
'''


from HW1_Problem_3 import *
from statistics import mode

CWD = os.getcwd()
PATH = CWD + "/mnist.npz"

def get_accuracy(vectors_distances, labels, k = 5, cosine_distances = True):
    '''
    Function -- get_accuracy
    Calculate the classification accuracy of training and testing KNN 
    Parameters: vectors_distances (matrix) -- vectors' distances matrix
                labels (array) -- an array of instances' labels
                k (int) -- number of neighbors in the KNN model
                cosine_distances (bool) -- applied cosine distances or not
    Return the accuracy number representing model's performance
    '''
    if cosine_distances:
        # Get the indices of each instance's k nearest neighbors
        knn_indices = vectors_distances.argsort(axis = 1)[:, -k:]
    else:
        knn_indices = vectors_distances.argsort(axis = 1)[:, :k]
    # Predict instance's label based on majority vote
    predictions = np.array([mode(labels[knn_indices[i]]) for i in range(len(labels))])
    # Calculate the classification accuracy
    accuracy = round(sum(predictions == labels) / len(labels), 4)
    return accuracy

def main():
    print("Loading datasets...")
    mnist_train, mnist_train_labels, mnist_test, mnist_test_labels = parse_MNIST(PATH)
    mnist_train, mnist_train_labels = mnist_train[:10000], mnist_train_labels[:10000]
    ng_dataset = fetch_20newsgroups(remove = ("headers", "footers", "quotes"),
                                    subset = "train")
    ng_train, ng_train_labels = ng_dataset.data, ng_dataset.target
    ng_testset = fetch_20newsgroups(remove = ("headers", "footers", "quotes"),
                                    subset = "test")
    ng_test, ng_test_labels = ng_testset.data, ng_testset.target
    print("Loading complete.\n")
    print("20NG datasets training performance:")
    vectors = parse_20_NG(ng_train, max_features = 5000, use_idf = True)
    vectors_distances = get_cosine_distances(vectors)
    accuracy = get_accuracy(vectors_distances, ng_train_labels,
                            k = 5, cosine_distances = True)
    print(accuracy)
    print("\n20NG datasets testing performance:")
    vectors = parse_20_NG(ng_test, max_features = 5000, use_idf = True)
    vectors_distances = get_cosine_distances(vectors)
    accuracy = get_accuracy(vectors_distances, ng_test_labels,
                            k = 5, cosine_distances = True)
    print(accuracy)
    print("\nMNIST datasets training performance:")
    vectors = mnist_train
    vectors_distances = edit_distances_MNIST(vectors)
    accuracy = get_accuracy(vectors_distances, mnist_train_labels,
                            k = 5, cosine_distances = False)
    print(accuracy)
    print("\nMNIST datasets testing performance:")
    vectors = mnist_test
    vectors_distances = edit_distances_MNIST(vectors)
    accuracy = get_accuracy(vectors_distances, mnist_test_labels,
                            k = 5, cosine_distances = False)
    print(accuracy)


if __name__ == "__main__":
    main()
