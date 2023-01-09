'''
    DS 5230
    Summer 2022
    HW3B_Problem_1_Pairwise_feature_selection_for_text
    HW3B_Problem_2_L1_feature_selection_on_text

    Train and test feature selection using skikit-learn built in "chi2"
    criteria and "mutual-information" criteria on the 20NG dataset
   
    Train and test L1-reg Logistic Regression on the 20NG datasets and
    select 200 features based on regression coefficients absolute value
    Analyze the model trained in terms of features

    Hongyan Yang
'''


from HW3_Problem_1 import *
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif as mi

def map_class_name(class_name_list, feature_names):
    '''
    Map real news group names to selected NG features
    '''
    class_dict = dict()
    for i in range(len(class_name_list)):
        class_dict[class_name_list[i]] = feature_names[i]
    return class_dict

def main():
    print("### A) Problem 1. Pairwise feature selection for text\n")
    ng_set = fetch_20newsgroups(remove = ("headers", "footers", "quotes"),
                                subset = "all")
    ng_data, names = parse_20_NG(ng_set.data, max_features = 5000, use_idf = False)
    data = train_test_split(ng_data, ng_set.target, test_size = 0.25,
                            random_state = 0)
    train, test, train_labels, test_labels = data
    print("## Run L2-reg Logistic Regression with all features:\n")
    clf = LogisticRegression(penalty = "l2", solver = "lbfgs", tol = 1e0,
                             max_iter = 800)
    clf.fit(train, train_labels)
    score = clf.score(test, test_labels)
    print(f"# Purity: {score}\n")
    print("## Run feature selection using 'chi2' criteria:\n")
    chi2_indices = np.argsort(chi2(train, train_labels)[0])[::-1][:200]
    train_chi2, test_chi2 = train[:, chi2_indices], test[:, chi2_indices]
    clf.fit(train_chi2, train_labels)
    score_chi2 = clf.score(test_chi2, test_labels)
    print(f"# Purity: {score_chi2}\n")
    print("## Run feature selection using 'mutual-information:' criteria\n")
    mi_indices = np.argsort(mi(train, train_labels))[::-1][:200]
    train_mi, test_mi = train[:, mi_indices], test[:, mi_indices]
    clf.fit(train_mi, train_labels)
    score_mi = clf.score(test_mi, test_labels)
    print(f"# Purity: {score_mi}\n")
    print()
    print("### B) Problem 2. L1 feature selection on text\n")
    print("## Run L1-reg Logistic Regression with all features:\n")
    clf = LogisticRegression(penalty = "l1", solver = "saga", tol = 1e0,
                             max_iter = 800)
    clf.fit(train, train_labels)
    score = clf.score(test, test_labels)
    print(f"# Purity: {score}\n")
    print("## Run L1-reg Logistic Regression with selected 200 features:\n")
    print("# The highest-absolute-value 200 coefficients are as follows:\n")
    coefficients_indices = np.argsort(-np.absolute(clf.coef_), axis = 1)[:, :10]
    feature_names = [names[i] for i in coefficients_indices]
    class_name_list = ng_set.target_names
    class_dict = map_class_name(class_name_list, feature_names)
    for i in range(3):
        key = list(class_dict.keys())[i]
        features = class_dict[key]
        print(f"NEWS GROUP {key}:\n")
        print(features)
        print()
    l1_indices = np.unique(coefficients_indices)
    train_l1, test_l1 = train[:, l1_indices], test[:, l1_indices]
    clf.fit(train_l1, train_labels)
    score_l1 = clf.score(test_l1, test_labels)
    print(f"# Purity: {score_l1}")

    
if __name__ == "__main__":
    main()
