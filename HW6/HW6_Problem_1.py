'''
    DS 5230
    Summer 2022
    HW6_Problem_1_Recommender_System_using_Collaborative_Filtering

    Implement a Movie Recommendation System, run it on the Movie Lens Datase

    Hongyan Yang
'''


import os
import numpy as np
import pandas as pd

from numpy import isnan

CWD = os.getcwd()
FILE_PATH = CWD + "/ml-100k"

def get_normalize_m(vectors):
    '''
    Return normalized matrix with each row mu equals zero, variance equals one
    '''
    vectors_mean = np.nanmean(vectors, axis=1).reshape(-1, 1)
    vectors_std = np.nanstd(vectors, axis=1).reshape(-1, 1)
    out_vectors = (vectors - vectors_mean) / vectors_std
    return out_vectors

def get_similarity(vectors):
    '''
    Return a similarity matrix based on the Pearson's Formula
    '''
    # Get the number of movies rated by users in common
    count_m = np.copy(vectors)
    count_m[~isnan(count_m)] = 1
    count_m = np.nan_to_num(count_m)
    count_intersect = np.matmul(count_m, count_m.T)
    count_intersect[count_intersect == 0] = np.inf
    # Get the sum of multiple of ratings
    mul_m = np.copy(vectors)
    mul_m = np.nan_to_num(mul_m)
    rating_intersect = np.matmul(mul_m, mul_m.T)
    # Get the similarity matrix
    similarity_m = rating_intersect / count_intersect
    return similarity_m

def get_predict(user_ids, movie_ids, user_m, vectors, similarity_m):
    '''
    Predict the rating for a movie by accounting for all other users' rating
    '''
    mu_u = np.nanmean(user_m[user_ids - 1], axis=1)
    sigma_u = np.nanstd(user_m[user_ids - 1], axis=1)
    # Get user's similarity with others
    similarity = similarity_m[user_ids - 1]
    # Get others' ratings, set user's rating to nan
    ratings = vectors[:, movie_ids - 1]
    ratings[user_ids - 1, range(ratings.shape[1])] = np.nan
    # Get others' weighted ratings as the predict rating
    rho_sum = np.stack([np.sum(np.abs(similarity[i][~isnan(ratings.T[i])]))
                        for i in range(len(similarity))])
    rho_sum[rho_sum == 0] = np.inf
    ratings_adj = np.nan_to_num(ratings)
    weighted_rating = np.stack([np.dot(similarity[i], ratings_adj.T[i])
                                for i in range(len(similarity))])
    predict_ratings = mu_u + weighted_rating / rho_sum * sigma_u
    return predict_ratings

def main():
    print("# A) Load the datasets.\n")
    # Read training dataframe
    col_names = ["user_id", "item_id", "rating", "timestamp"]
    ua_df = pd.read_csv(FILE_PATH + "/ua.base", sep="\t", names=col_names,
                        encoding="utf_8")
    # Read test dataframe
    ua_t_df = pd.read_csv(FILE_PATH + "/ua.test", sep="\t", names=col_names)
    # Read user dataframe
    col_names = ["user_id", "age", "gender", "occupation", "zip_code"]
    user_df = pd.read_csv(FILE_PATH + "/u.user", sep="|", names=col_names,
                          encoding="utf_8")
    # Read user dataframe
    col_names = ["movie_id", "movie_title", "release_date", "video_release_date",
                 "IMDb_URL", "unknown", "Action", "Adventure", "Animation",
                 "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
                 "Film_Noir", "Horror", "Musical", "Mystery", "Romance", "Sci_Fi",
                 "Thriller", "War", "Western"]
    item_df = pd.read_csv(FILE_PATH + "/u.item", sep="|", names=col_names,
                          encoding="mbcs", encoding_errors="ignore")

    print("# B) Predict user's ratings and report RMSE.\n")
    # Create user_matrix
    len_row = user_df.shape[0]
    len_col = item_df.shape[0]
    user_m = np.empty((len_row, len_col))   # Create an empty matrix
    user_m.fill(np.nan)
    for index, row in ua_df.iterrows():
        user_m[row["user_id"] - 1, row["item_id"] - 1] = row["rating"]
    # Make predictions
    vectors = get_normalize_m(user_m)
    similarity_m = get_similarity(vectors)
    user_ids = ua_t_df["user_id"]
    movie_ids = ua_t_df["item_id"]
    true_ratings = ua_t_df["rating"]
    predict_ratings = get_predict(user_ids, movie_ids, user_m, vectors, similarity_m)
    MSE = np.mean(np.square(predict_ratings - true_ratings))
    RMSE = np.sqrt(MSE)
    print(RMSE)
        

if __name__ == "__main__":
    main()
