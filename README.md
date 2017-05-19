# Content-Based-Movie-Recommendation-System

Here we'll implement a Content Based Recommendation System. 
The content here will be list of genres for a movie. 
The data we are using is taken from MovieLens Project.

We use the tf-idf approach which is short for term frequency–inverse document frequency, is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus. It is often used as a weighting factor in information retrieval and text mining.

After featurizing and vectorizing using tfidf we now find similarity between movies user has rated and the new movie to be recommended and we used cosine similarity for this purpose.

#### To predict the rating of user u for movie i: 
    We compute the weighted average rating for every other movie that u has rated.
    Restrict this weighted average to movies that have a positive cosine similarity with movie i. 
    The weight for movie m corresponds to the cosine similarity between m and i. 
    If there are no other movies with positive cosine similarity to use in the prediction, 
    we use the mean rating of the target user in ratings_train as the prediction.
