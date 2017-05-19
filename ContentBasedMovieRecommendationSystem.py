
# Content based Movie Recommendation System

from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile


def tokenize_string(my_string):

    return re.findall('[\w\-]+', my_string.lower())


def tokenize(movies):
    """
    Append a new column to the movies DataFrame with header 'tokens'.
    This will contain a list of strings, one per token, extracted
    from the 'genre' field of each movie. 

    Params:
      movies...The movies DataFrame
    Returns:
      The movies DataFrame, augmented to include a new column called 'tokens'.

    """
    movies['tokens'] = movies['genres'].map(lambda x: tokenize_string(x))    
    return movies

def featurize(movies):
    """
    Append a new column to the movies DataFrame with header 'features'.
    Each row will contain a csr_matrix of shape (1, num_features). Each
    entry in this matrix will contain the tf-idf value of the term.

    tfidf(i, d) := tf(i, d) / max_k tf(k, d) * log10(N/df(i))
    where:
    i is a term
    d is a document (movie)
    tf(i, d) is the frequency of term i in document d
    max_k tf(k, d) is the maximum frequency of any term in document d
    N is the number of documents (movies)
    df(i) is the number of unique documents containing term i

    Params:
      movies...The movies DataFrame
    Returns:
      A tuple containing:
      - The movies DataFrame, which has been modified to include a column named 'features'.
      - The vocab, a dict from term to int. Make sure the vocab is sorted alphabetically as in a2 (e.g., {'aardvark': 0, 'boy': 1, ...})
    """
    vocab_set = set()
    
    for movie in movies['tokens']:
        for token in movie:
            vocab_set.add(token)
    
    vocab_list = list(vocab_set)
    
    df = Counter()
    
    for v in vocab_list:
        for movie in movies['tokens']:
            if v in movie:
                df[v] += 1
    
    vocab_list.sort()
    feature_list = []
    vocab = defaultdict()
    for index,value in enumerate(vocab_list):
        vocab[value] = index

    N = movies.shape[0]
    for movie in movies['tokens']:
        tf = Counter()
        for tok in movie:
            tf[tok] += 1
        
        max_k = sorted(tf.values(), key=lambda x: -x)[0]
        data = []
        col = []
        row = []
        for tok in tf:
            if tok in vocab:
                tfid = ( (tf[tok]/max_k) * (math.log10(N/df[tok])) )

                col.append(vocab[tok])
                data.append(tfid)
                row.append(0)

        
        feature_list.append(csr_matrix((data, (row, col)), shape=(1, len(vocab))).toarray())
    movies['features'] = feature_list
    return (movies,vocab)


def train_test_split(ratings):
    """
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]


def cosine_sim(a, b):
    """
    Compute the cosine similarity between two 1-d csr_matrices.
    Each matrix represents the tf-idf feature vector of a movie.
    Params:
      a...A csr_matrix with shape (1, number_features)
      b...A csr_matrix with shape (1, number_features)
    Returns:
      The cosine similarity, defined as: dot(a, b) / ||a|| * ||b||
      where ||a|| indicates the Euclidean norm (aka L2 norm) of vector a.
    """

    return (np.dot(a,np.transpose(b))/(np.linalg.norm(a)*np.linalg.norm(b)))[0][0]



def make_predictions(movies, ratings_train, ratings_test):
    """
    Using the ratings in ratings_train, predict the ratings for each
    row in ratings_test.

    To predict the rating of user u for movie i: Compute the weighted average
    rating for every other movie that u has rated.  Restrict this weighted
    average to movies that have a positive cosine similarity with movie
    i. The weight for movie m corresponds to the cosine similarity between m
    and i.

    If there are no other movies with positive cosine similarity to use in the
    prediction, use the mean rating of the target user in ratings_train as the
    prediction.

    Params:
      movies..........The movies DataFrame.
      ratings_train...The subset of ratings used for making predictions. These are the "historical" data.
      ratings_test....The subset of ratings that need to predicted. These are the "future" data.
    Returns:
      A numpy array containing one predicted rating for each element of ratings_test.
    """
    ratings_test_subset = ratings_test[['userId','movieId']]
    ratings_train_subset = ratings_train[['userId','movieId','rating']]
    
    predicted_ratings = []
    
    for test_row in ratings_test_subset.itertuples():
        
        ratings_train_subset_forUserId = ratings_train_subset[ratings_train_subset['userId'] == test_row[1]]
        weighted_rating = 0.0
        size = ratings_train_subset_forUserId.shape[0]
        rating_sum = 0.0
        cos_sum = 0.0

        b = movies[movies['movieId'] == test_row[2]]['features'].values[0]

        for train_row in ratings_train_subset_forUserId.itertuples():
            a = movies[movies['movieId'] == train_row[2]]['features'].values[0]
            sim = cosine_sim(a, b)

            rate = train_row[3]
            
            rating_sum = rating_sum + rate
            weighted_rating = weighted_rating + (sim * rate)
            cos_sum = cos_sum + sim

        
        if weighted_rating > 0:
            predicted_ratings.append(weighted_rating/cos_sum)
        else:
            predicted_ratings.append(rating_sum/size)
        
            
    return (np.array(predicted_ratings))
    



def mean_absolute_error(predictions, ratings_test):
    """
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()


def main():
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])


if __name__ == '__main__':
    main()

