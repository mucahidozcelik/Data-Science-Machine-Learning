#############################
# Content Based Recommendation
#############################

#############################
# Film Overview'larına Göre Tavsiye Sistemi
#############################

# 1. TF-IDF Matrisinin Oluşturulması
# 2. Cosine Similarity Matrisinin Oluşturulması
# 3. Benzerliklere Göre Önerilerin Yapılması
# 4. Çalışma Scriptinin Hazırlanması



#################################
# 1. TF-IDF Matrisinin Oluşturulması
#################################

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("datasets/the_movies_dataset/movies_metadata.csv", low_memory=False)  # DtypeWarning kapamak icin
df.head()

df["overview"].head()


# countVectorizer yöntemi
# tf-idf yöntemi



#################################
# countVectorizer yöntemi
#################################


# Count işleminde her bir kelimenin her bir dokümanda kaç defa geçtiği sayılır.


from sklearn.feature_extraction.text import CountVectorizer
corpus = ['This is the first document.',
          'This document is the second document.',
          'And this is the third one.',
          'Is this the first document?']



vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
vectorizer.get_feature_names()
X.toarray()


#################################
# tf-idf yöntemi
#################################


# TF-IDF = TF(t) * IDF(t)

# ADIM 1: TF(t) = (Bir t teriminin ilgili dokümanda gözlenme frekansı) / (Dokümandaki toplam terim sayısı) (term frequency)
# ADIM 2: IDF(t) = 1 + log_e((Toplam doküman sayısı + 1) / (İçinde t terimi olan doküman sayısı + 1) (inverse document frequency)
# ADIM 3: TF-IDF = TF(t) * IDF(t)
# ADIM 4: TF-IDF Değerlerine L2 normalization.


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(analyzer='word')
X = vectorizer.fit_transform(corpus)
vectorizer.get_feature_names()

X.toarray()


#################################
# TF-IDF'in Problemimiz için Elde Edilmesi
#################################

df['overview'].head()

tfidf = TfidfVectorizer(stop_words='english')
df['overview'] = df['overview'].fillna('')
tfidf_matrix = tfidf.fit_transform(df['overview'])
tfidf_matrix.shape

#################################
# 2. Cosine Similarity Matrisinin Oluşturulması
#################################


cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

cosine_sim.shape

cosine_sim[1]


#################################
# 3. Benzerliklere Göre Önerilerin Yapılması
#################################


indices = pd.Series(df.index, index=df['title'])

indices = indices[~indices.index.duplicated(keep='last')]

indices.shape
indices[:10]

indices["Sherlock Holmes"]

movie_index = indices["Sherlock Holmes"]


cosine_sim[movie_index]

similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["score"])

movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index

df['title'].iloc[movie_indices]


#################################
# 4. Çalışma Scriptinin Hazırlanması
#################################

def content_based_recommender(title, cosine_sim, dataframe):
    # index'leri olusturma
    indices = pd.Series(dataframe.index, index=dataframe['title'])
    indices = indices[~indices.index.duplicated(keep='last')]
    # title'ın index'ini yakalama
    movie_index = indices[title]
    # title'a gore benzerlik skorlarını hesapalama
    similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["score"])
    # kendisi haric ilk 10 filmi getirme
    movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index
    return dataframe['title'].iloc[movie_indices]

content_based_recommender("Sherlock Holmes", cosine_sim, df)

content_based_recommender("The Matrix", cosine_sim, df)

content_based_recommender("The Godfather", cosine_sim, df)


content_based_recommender('The Dark Knight Rises', cosine_sim, df)

del cosine_sim

content_based_recommender('The Dark Knight Rises', cosine_sim, df)


def calculate_cosine_sim(dataframe):
    tfidf = TfidfVectorizer(stop_words='english')
    dataframe['overview'] = dataframe['overview'].fillna('')
    tfidf_matrix = tfidf.fit_transform(dataframe['overview'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim


cosine_sim = calculate_cosine_sim(df)

content_based_recommender('The Dark Knight Rises', cosine_sim, df)