'''
    DS 5230
    Summer 2022
    HW5A_Problem_2_Topic_Models

    Obtain Topic Models (K = 10, 20, 50) by running LDA and NMF methods

    Hongyan Yang
'''


from HW5A_Problem_1 import *
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA

DUC2001_PATH = CWD + "/DUC2001/files"
PUNCTUATIONS = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ '

os.chdir(DUC2001_PATH)
warnings.filterwarnings("ignore")
                     
def plot_top_words(model, feature_names, title, n_top_words = 20):
    '''
    Plot topic models and plot each topic with the top 20 words
    '''
    topic_dict = dict()
    k = len(model.components_)
    n_row = int(k / 5) + (k % 5 > 0)
    fig, axes = plt.subplots(n_row, 5, sharex = True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        if k > 20:
            top_features_ind = topic.argsort()[: -9 : -1]
        else:
            top_features_ind = topic.argsort()[: (-n_top_words - 1) : -1]
        top_features = [feature_names[i] for i in top_features_ind]
        try:
            weights = topic[top_features_ind] / np.sum(topic)
            #weights = topic[top_features_ind]
        except:
            weights = np.zeros(len(topic))
        probs = np.round_(weights, 4).tolist()
        topic_dict[f"Topic {topic_idx +1}"] = [top_features, probs]
        # Set axes parameters
        ax = axes[topic_idx]
        ax.barh(top_features, weights)
        if k <= 20:
            ax.set_title(f"Topic {topic_idx +1}")
        ax.invert_yaxis()
        ax.tick_params(axis = "both", which = "major")
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
    if k <= 20:
        fig.suptitle(title)
    plt.show()
    return topic_dict

def extract_content(file_path):
    '''
    Extract main body content from the DUC2001 dataset files
    '''
    with open(file_path, "r", encoding = "utf-8") as f:
        file = f.read()
        start, end = file.find("<TEXT>"), file.find("</TEXT>")
        content = file[start + 6: end]
        content = content.replace("<P>", "").replace("</P>", "")
        content = content.replace("\n", " ").replace("-", " ")
        content_list = [each.strip(PUNCTUATIONS).lower()
                        for each in content.split()]
        while ("" in content_list):
            content_list.remove("")
        content = " ".join(content_list)
        return content

def get_contents(path = DUC2001_PATH):
    '''
    Compile all main body content from the DUC2001 dataset files
    '''
    contents_list = list()
    for file in os.listdir():
        file_path = path + f"/{file}"
        content = extract_content(file_path)
        contents_list.append(content)
    return contents_list

def main():
    print("### A) Run NMF on the DUC2001 dataset\n")
    duc_set = get_contents(DUC2001_PATH)
    # Use tf-idf features for the NMF model
    print("## Extracte tf-idf features for the NMF model\n")
    vct = TfidfVectorizer(decode_error = "replace", max_df = 0.95, min_df = 2,
                          stop_words = "english", use_idf = True)
    vectors = vct.fit_transform(duc_set)
    names = vct.get_feature_names_out()
    title = "Topics in NMF model for the DUC2001 dataset"
    print("# Topic Model with K = 10\n")
    nmf = NMF(n_components = 10, init = "nndsvda", alpha_W = 0.00005,
              alpha_H = 0.00005, l1_ratio = 1).fit(vectors)
    tpc = plot_top_words(nmf, names, title)
    print(tpc["Topic 1"][0])
    print(tpc["Topic 1"][1])
    print()
    print("# Topic Model with K = 20\n")
    nmf = NMF(n_components = 20, init = "nndsvda", alpha_W = 0.00005,
              alpha_H = 0.00005, l1_ratio = 1).fit(vectors)
    tpc = plot_top_words(nmf, names, title)
    print(tpc["Topic 1"][0])
    print(tpc["Topic 1"][1])
    print()
    print("# Topic Model with K = 50\n")
    nmf = NMF(n_components = 50, init = "nndsvda", alpha_W = 0.00005,
              alpha_H = 0.00005, l1_ratio = 1).fit(vectors)
    tpc = plot_top_words(nmf, names, title)
    print(tpc["Topic 1"][0])
    print(tpc["Topic 1"][1])
    print("\n")

    print("### B) Run LDA on the DUC2001 dataset\n")
    # Use tf features for the LDA model
    print("## Extract tf features for the LDA model\n")
    vct = CountVectorizer(max_df = 0.95, min_df = 2,
                          max_features = 1000, stop_words = "english")
    vectors = vct.fit_transform(duc_set)
    names = vct.get_feature_names_out()
    title = "Topics in LDA model for the DUC2001 dataset"
    print("# Topic Model with K = 10\n")
    lda = LDA(n_components = 10, max_iter = 5, learning_method = "online",
              learning_offset = 50.0).fit(vectors)
    tpc = plot_top_words(lda, names, title)
    print(tpc["Topic 1"][0])
    print(tpc["Topic 1"][1])
    print()
    print("# Topic Model with K = 20\n")
    lda = LDA(n_components = 20, max_iter = 5, learning_method = "online",
              learning_offset = 50.0).fit(vectors)
    tpc = plot_top_words(lda, names, title)
    print(tpc["Topic 1"][0])
    print(tpc["Topic 1"][1])
    print()
    print("# Topic Model with K = 50\n")
    lda = LDA(n_components = 50, max_iter = 5, learning_method = "online",
              learning_offset = 50.0).fit(vectors)
    tpc = plot_top_words(lda, names, title)
    print(tpc["Topic 1"][0])
    print(tpc["Topic 1"][1])
    print("\n")

    print("### C) Run NMF on the 20NG dataset\n")
    ng_set = fetch_20newsgroups(remove = ("headers", "footers", "quotes"),
                                subset = "test")
    # Use tf-idf features for the NMF model
    print("## Extracte tf-idf features for the NMF model\n")
    vectors, names = parse_20_NG(ng_set.data, max_features = 1000,
                                 use_idf = True)
    title = "Topics in NMF model for the 20NG dataset"
    print("# Topic Model with K = 10\n")
    nmf = NMF(n_components = 10, init = "nndsvda", alpha_W = 0.00005,
              alpha_H = 0.00005, l1_ratio = 1).fit(vectors[:2000])
    tpc = plot_top_words(nmf, names, title)
    print(tpc["Topic 1"][0])
    print(tpc["Topic 1"][1])
    print()
    print("# Topic Model with K = 20\n")
    nmf = NMF(n_components = 20, init = "nndsvda", alpha_W = 0.00005,
              alpha_H = 0.00005, l1_ratio = 1).fit(vectors[:2000])
    tpc = plot_top_words(nmf, names, title)
    print(tpc["Topic 1"][0])
    print(tpc["Topic 1"][1])
    print()
    print("# Topic Model with K = 50\n")
    nmf = NMF(n_components = 50, init = "nndsvda", alpha_W = 0.00005,
              alpha_H = 0.00005, l1_ratio = 1).fit(vectors[:2000])
    tpc = plot_top_words(nmf, names, title)
    print(tpc["Topic 1"][0])
    print(tpc["Topic 1"][1])
    print("\n")

    print("### D) Run LDA on the 20NG dataset\n")
    # Use tf features for the LDA model
    print("## Extracte tf features for the LDA model\n")
    vct = CountVectorizer(max_df = 0.95, min_df = 2,
                          max_features = 1000, stop_words = "english")
    vectors = vct.fit_transform(ng_set.data)
    names = vct.get_feature_names_out()
    title = "Topics in LDA model for the 20NG dataset"
    print("# Topic Model with K = 10\n")
    lda = LDA(n_components = 10, max_iter = 5, learning_method = "online",
              learning_offset = 50.0).fit(vectors[:2000])
    tpc = plot_top_words(lda, names, title)
    print(tpc["Topic 1"][0])
    print(tpc["Topic 1"][1])
    print()
    print("# Topic Model with K = 20\n")
    lda = LDA(n_components = 20, max_iter = 5, learning_method = "online",
              learning_offset = 50.0).fit(vectors[:2000])
    tpc = plot_top_words(lda, names, title)
    print(tpc["Topic 1"][0])
    print(tpc["Topic 1"][1])
    print()
    print("# Topic Model with K = 50\n")
    lda = LDA(n_components = 50, max_iter = 5, learning_method = "online",
              learning_offset = 50.0).fit(vectors[:2000])
    tpc = plot_top_words(lda, names, title)
    print(tpc["Topic 1"][0])
    print(tpc["Topic 1"][1])
    print("\n")

    print("### Complete")

if __name__ == "__main__":
    main()
