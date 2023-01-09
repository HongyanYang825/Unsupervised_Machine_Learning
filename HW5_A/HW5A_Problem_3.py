'''
    DS 5230
    Summer 2022
    HW5A_Problem_3_Extractive_Summarization

    Implement the KL-Sum summarization method for the 20NG and DUC2001
    datasets
    Evaluate KL_summaries and LDA_summaries against human gold summaries
    with ROUGE

    Hongyan Yang
'''


import math
import functools

from HW5A_Problem_2 import *
from nltk.corpus import stopwords
from collections import defaultdict
from rouge_score import rouge_scorer

KL_PATH = CWD + "/DUC2001/KL_Summaries"
LDA_PATH = CWD + "/DUC2001/LDA_Summaries"
GOLD_PATH = CWD + "/DUC2001/Summaries"

CHARS_TO_REMOVE = [",", ";", "'s", "@", "&", "*", "(", ")", "#", "!",
                   "%", "=", "+", "-", "_", ":", '"', "'"]
CHARS_TO_REPLACE = ["?"]
STOPWORDS = set(stopwords.words("english"))

def get_sentences(file_path):
    '''
    Extract all sentences from one DUC2001 dataset file
    '''
    with open(file_path, "r", encoding = "utf-8") as f:
        file = f.read()
        start, end = file.find("<TEXT>"), file.find("</TEXT>")
        content = file[start + 6: end]
        content = content.replace("<P>", "").replace("</P>", "")
        content = content.replace("\n", " ").replace("-", " ")
        for item in CHARS_TO_REMOVE:
            content = content.replace(item, "")
        for item in CHARS_TO_REPLACE:
            content = content.replace(item, ".")
        # Extract all sentences by "."
        sentences_list = [each.strip(PUNCTUATIONS).lower()
                          for each in content.split(".")]
        c_s = lambda x: " ".join(filter(None, x.split(" ")))
        sentences_list = list(map(c_s, sentences_list))
        # Filter the stopwords
        f_s = lambda x: " ".join(filter(lambda y: y not in STOPWORDS,
                                        x.split(" ")))
        filtered_sentences = list(map(f_s, sentences_list))
        filtered_cleaned = list(filter(None, filtered_sentences))
        return filtered_cleaned, filtered_sentences, sentences_list

def get_sentences_20ng(file):
    '''
    Extract all sentences from one 20NG dataset file
    '''
    file = file.replace("\n", " ").replace("-", " ").replace("\t", " ")
    for item in CHARS_TO_REMOVE:
        file = file.replace(item, "")
    for item in CHARS_TO_REPLACE:
        file = file.replace(item, ".")
    sentences_list = [each.strip(PUNCTUATIONS).lower()
                      for each in file.split(".")]
    c_s = lambda x: " ".join(filter(None, x.split(" ")))
    sentences_list = list(map(c_s, sentences_list))
    f_s = lambda x: " ".join(filter(lambda y: y not in STOPWORDS,
                                    x.split(" ")))
    filtered_sentences = list(map(f_s, sentences_list))
    filtered_cleaned = list(filter(None, filtered_sentences))
    return filtered_cleaned, filtered_sentences, sentences_list

def get_doc_words(sentences_list, n_gram = 1):
    '''
    Extract all words from one DUC2001 dataset file
    '''
    sentences_to_words = map(lambda x: x.split(), sentences_list)
    to_n_gram = lambda x: zip(*[x[i:] for i in range(n_gram)])
    zip_list = map(to_n_gram, sentences_to_words)
    to_words = lambda x: [" ".join(y) for y in x]
    n_gram_list = map(to_words, zip_list)
    words_list = list(functools.reduce(lambda x, y: x + y,
                                       n_gram_list))
    return words_list

def get_s_words(sentence, n_gram = 1):
    '''
    Extract all words of one sentence from a DUC2001 dataset file
    '''
    n_gram = zip(*[sentence.split()[i:] for i in range(n_gram)])
    words_list = [" ".join(x) for x in n_gram]
    return words_list

def get_term_freq(words_list):
    '''
    Compute term frequency of a list of words
    '''
    word_freq_dict = defaultdict(int)
    for word in words_list:
        word_freq_dict[word] += 1
    total_wf = len(words_list)
    tf_dict = dict((w, f / total_wf) for w, f in word_freq_dict.items())
    return tf_dict

def get_kl_divergence(summary_freq, doc_freq):
    '''
    Calculate the KL divergence between one sentence and the document
    The lower the value the better
    '''
    sum_val = 0
    for word in summary_freq:
        freq = doc_freq.get(word)
        if freq:  # False if the word not exist in doc_freq
            sum_val += freq * math.log(freq / summary_freq[word])
    return sum_val

def get_kl_divegence_lda(p_s, p_d):
    '''
    Calculate the KL divergence between one sentence and the document
    The lower the value the better
    '''
    val_array = np.hstack([np.sum(p_d[0] * np.log(p_d[0] / p_s[i]))]
                          for i in range(len(p_s)))
    return val_array.tolist()

def get_summary(file_path, n_gram = 1, word_limit = 150,
                NG = False, file = None):
    '''
    Generate the summary for a given document, stopwords are filled to
    complete the sentences
    '''
    # Get term frequency for the document
    if NG:
        sentences = get_sentences_20ng(file)
    else:
        sentences = get_sentences(file_path)
    doc_words_list = get_doc_words(sentences[0], n_gram = n_gram)
    doc_tf = get_term_freq(doc_words_list)
    # Get term frequency for sentence
    kl_list = list()
    for sentence in sentences[0]:
        s_words_list = get_s_words(sentence, n_gram = n_gram)
        s_tf = get_term_freq(s_words_list)
        kl_value = get_kl_divergence(s_tf, doc_tf)
        kl_list.append(kl_value)
    # Add sentence to summary until length hit word limit
    summary_list, index_list, length = list(), list(), 0
    kl_copy = kl_list[:]
    while True:
        orig_index = kl_list.index(min(kl_copy))
        c_s = sentences[0][orig_index]
        full_s = sentences[2][sentences[1].index(c_s)]
        kl_copy.remove(min(kl_copy))
        length += len(full_s.split())
        if length > word_limit or len(kl_copy) == 0:
            break
        if orig_index in index_list:
            continue
        summary_list.append(full_s)
        index_list.append(orig_index)
    sorted_s = sorted(zip(summary_list, index_list),
                      key = lambda x: x[1])
    summary_s = [s[0] + ". " for s in sorted_s]
    try:
        summary = str(functools.reduce(lambda x, y: x + y,
                                       summary_s))
    except:
        return ""
    return summary.strip()

def get_summary_lda(file_path, n_topic = 20, word_limit = 150,
                    NG = False, file = None):
    '''
    Apply LDA to generate the summary for a given document, stopwords are
    filled to complete the sentences
    '''
    vct = CountVectorizer()
    lda = LDA(n_components=n_topic, max_iter=5, learning_method="online",
              learning_offset=50.0)
    # Get topic distribution for the document
    if NG:
        sentences = get_sentences_20ng(file)
    else:
        sentences = get_sentences(file_path)
    doc_str = functools.reduce(lambda x, y: x + " " + y, sentences[0])
    doc_tf = vct.fit_transform([doc_str])
    p_d = lda.fit_transform(doc_tf)
    # Get topic distribution for sentences
    s_tf = vct.fit_transform(sentences[0])
    p_s = lda.transform(s_tf)
    # Get the KL divergence between sentences and the document
    kl_list = get_kl_divegence_lda(p_s, p_d)
    # Add sentence to summary until length hit word limit
    summary_list, index_list, length = list(), list(), 0
    kl_copy = kl_list[:]
    while True:
        orig_index = kl_list.index(min(kl_copy))
        c_s = sentences[0][orig_index]
        full_s = sentences[2][sentences[1].index(c_s)]
        kl_copy.remove(min(kl_copy))
        length += len(full_s.split())
        if length > word_limit or len(kl_copy) == 0:
            break
        if orig_index in index_list:
            continue
        summary_list.append(full_s)
        index_list.append(orig_index)
    sorted_s = sorted(zip(summary_list, index_list),
                      key = lambda x: x[1])
    summary_s = [s[0] + ". " for s in sorted_s]
    try:
        summary = str(functools.reduce(lambda x, y: x + y,
                                       summary_s))
    except:
        return ""
    return summary.strip()

def write_summaries_20ng():
    ng = fetch_20newsgroups(remove = ("headers", "footers", "quotes"),
                            subset = "all").data[:1000]
    ng = list(filter(lambda x: len(x) > 10, ng))
    kl_fun = lambda x: get_summary(None, n_gram = 1, word_limit = 50,
                                   NG = True, file = x)
    kl_summary = list(map(kl_fun, ng))
    lda_fun = lambda x: get_summary_lda(None, 20, 50, True, x)
    lda_summary = list(map(lda_fun, ng))
    return kl_summary, lda_summary

def write_summaries(path = DUC2001_PATH, kl_path = KL_PATH,
                    lda_path = LDA_PATH, lda = False):
    '''
    Write summaries for all files in the DUC2001 dataset
    '''
    for file in os.listdir():
        file_path = path + f"/{file}"
        if lda:
            write_path = lda_path + f"/{file}.txt"
            summary = ["\n" + get_summary_lda(file_path) + "\n"]
            with open(write_path, "w", encoding = "utf-8") as f:
                f.write("Summary:")
                f.writelines(summary)
        else:
            write_path = kl_path + f"/{file}.txt"
            summary = [("\n" + get_summary(file_path, n_gram = i)
                        + "\n")
                       for i in range(1, 5)]
            with open(write_path, "w", encoding = "utf-8") as f:
               f.write("Summary:")
               f.writelines(summary)

def extract_gold_summaries(path = GOLD_PATH):
    '''
    Extract human gold summaries for the DUC2001 dataset
    '''
    file_list, summary_list = list(), list()
    for file in os.listdir():
        file = file.lower()
        file_path = path + f"/{file}.txt"
        try:
            with open(file_path, "r", encoding = "utf-8") as f:
                title = f.readline()
                summary = f.readline()
                summary_list.append(summary.strip(" \n").lower())
                file_list.append(file)
        except:
            continue
    filter_fun = lambda x: len(summary_list[file_list.index(x)]) > 13
    file_list = list(filter(filter_fun, file_list))
    summary_list = list(filter(lambda x: len(x) > 13, summary_list))
    return file_list, summary_list

def parse_rouge(scorer, gold_summary, my_summary, length):
    '''
    Parse the rouge score object and return a dictionary
    '''
    score_dict = dict()
    scores = list(map(lambda a: scorer.score(a[0], a[1]),
                      zip(gold_summary, my_summary)))
    precision = round(sum([score['rouge1'].precision
                           for score in scores]) / length, 4)
    score_dict["precision"] = precision
    recall = round(sum([score['rouge1'].recall
                        for score in scores]) / length, 4)
    score_dict["recall"] = recall
    fmeasure = round(sum([score['rouge1'].fmeasure
                          for score in scores]) / length, 4)
    score_dict["fmeasure"] = fmeasure
    return score_dict

def get_rouge_score(file_list, summary_list, kl_path = KL_PATH,
                    lda_path = LDA_PATH):
    '''
    Evaluate KL_summaries and LDA_summaries against human gold summaries
    '''
    kl_1, kl_2, kl_3, kl_4 = [], [], [], []
    lda_summary = list()
    # Extract corresponding summaries from files' location
    for file in file_list:
        file = file.upper()
        kl_file = kl_path + f"/{file}.txt"
        lda_file = lda_path + f"/{file}.txt"
        with open(kl_file, "r", encoding = "utf-8") as f:
            title = f.readline()
            summary_1 = f.readline(); f.readline()
            summary_2 = f.readline(); f.readline()
            summary_3 = f.readline(); f.readline()
            summary_4 = f.readline();
            kl_1.append(summary_1)
            kl_2.append(summary_2)
            kl_3.append(summary_3)
            kl_4.append(summary_4)
        with open(lda_file, "r", encoding = "utf-8") as f:
            title = f.readline()
            summary = f.readline()
            lda_summary.append(summary)
    # Compute and average rouge scores of different summaries
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer = True)
    kl_1_dict = parse_rouge(scorer, summary_list, kl_1, len(file_list))
    kl_2_dict = parse_rouge(scorer, summary_list, kl_2, len(file_list))
    kl_3_dict = parse_rouge(scorer, summary_list, kl_3, len(file_list))
    kl_4_dict = parse_rouge(scorer, summary_list, kl_4, len(file_list))
    lda_dict = parse_rouge(scorer, summary_list, lda_summary, len(file_list))
    return kl_1_dict, kl_2_dict, kl_3_dict, kl_4_dict, lda_dict

def main():
    print("### A) Run KL_summary and LDA_summary on the 20NG dataset\n")
    kl_summary, lda_summary = write_summaries_20ng()
    print(kl_summary[0])
    print(lda_summary[0])
    print()
    while True:
        index = int(input("Which summary would you like to check" \
                          f" (0-{len(kl_summary) - 1})? Enter -1 to exit. "))
        print()
        if index == -1:
            break
        else:
            print(kl_summary[index])
            print(lda_summary[index])
            print()
    print()
    print("### B) Run KL_summary and LDA_summary on the DUC2001 dataset\n")
    write_summaries(path = DUC2001_PATH, kl_path = KL_PATH,
                    lda_path = LDA_PATH, lda = False)
    write_summaries(path = DUC2001_PATH, kl_path = KL_PATH,
                    lda_path = LDA_PATH, lda = True)
    print("## Complete. Please check summaries location to see the results.\n")
    print()
    print("### C) Evaluate summaries performance against gold summaries\n")
    file_list, summary_list = extract_gold_summaries(path = GOLD_PATH)
    scores = get_rouge_score(file_list, summary_list, kl_path = KL_PATH,
                             lda_path = LDA_PATH)
    print("## KL_summary, n_gram = 1:\n")
    print(scores[0], "\n")
    print("## KL_summary, n_gram = 2:\n")
    print(scores[1], "\n")
    print("## KL_summary, n_gram = 3:\n")
    print(scores[2], "\n")
    print("## KL_summary, n_gram = 4:\n")
    print(scores[3], "\n")
    print("## LDA_summary: ")
    print(scores[4], "\n")
    print()
    print("### Complete")


if __name__ == "__main__":
    main()
