import gensim
from nltk import word_tokenize
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict


# Function to get the cosine similarity between a relation and query
# Note: Be sure to prepend the relation with ns:
word2vec_model = gensim.models.Word2Vec.load('word2vec_train_dev.dat')
def get_rel_score_word2vecbase(rel, query):
    if rel not in word2vec_model.wv:
        return 0.0
    words = word_tokenize(query.lower())
    w_embs = []
    for w in words:
        if w in word2vec_model.wv:
            w_embs.append(word2vec_model.wv[w])
    return np.mean(cosine_similarity(w_embs, [word2vec_model.wv[rel]]))


# Function to load the graph from file
def load_graph():
    # Preparing the graph
    graph = defaultdict(list)
    for line in open('graph'):
        line = eval(line[:-1])
        graph[line[0]].append([line[1], line[2]])
    return graph


# Function to load the queries from file
# Preparing the queries
def load_queries():
    queries = []
    for line in open('annotations', encoding="utf-8"):
        line = eval(line[:-1])
        queries.append(line)
    return queries

graph = load_graph()
queries = load_queries()

def find_answers(query, graph, threshold = 0.3, depth = 3):
    wave = 0
    index, question, start_entity = query[:3]
    layer = graph[start_entity]
    relations = ["ns:" + each[0] for each in layer]
    new_members = [each[1] for each in layer]
    sims = [get_rel_score_word2vecbase(rel, question) for rel in relations]
    ans_indices = [i for i, k in enumerate(sims) if k > threshold]
    answers = set()
    for index in ans_indices:
        answers.add(new_members[index])
    old_ans = answers
    while wave < depth:
        wave_ans = set()
        for entity in old_ans:
            layer = graph[entity]
            relations = ["ns:" + each[0] for each in layer]
            new_members = [each[1] for each in layer]
            sims = [get_rel_score_word2vecbase(rel, question) for rel in relations]
            ans_indices = [i for i, k in enumerate(sims) if k > threshold]
            for index in ans_indices:
                wave_ans.add(new_members[index])
        old_ans = wave_ans
        answers.update(wave_ans)
        wave += 1
    return answers

def get_real_answers(query):
    answers = query[5]
    answers_set = set()
    for answer in answers:
        answers_set.add(answer["AnswerArgument"])
    return answers_set

def get_f1_score(answers, real_answers):
    try:
        precision = len(answers.intersection(real_answers)) / len(answers)
    except:
        precision = 0
    recall = len(answers.intersection(real_answers)) / len(real_answers)
    try:
        f1_score = 2 * precision * recall / (precision + recall)
    except:
        f1_score = 0
    return f1_score

def main():
    f1_scores = list()
    for i in range(len(queries)):
        query = queries[i]
        real_answers = get_real_answers(query)
        answers = find_answers(query, graph, threshold = 0.3, depth = 1)
        f1_scores.append(get_f1_score(answers, real_answers))
    print(f1_scores[:5])
    print(sum(f1_scores) / len(f1_scores))


if __name__ == "__main__":
    main()
        
