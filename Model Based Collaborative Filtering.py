#############################
# Model Based Collaborative Filtering: Matrix Factorization
#############################

# Adım 1: Veri Setinin Hazırlanması
# Adım 2: Modelleme
# Adım 3: Model Tuning
# Adım 4: Final Model ve Tahmin


# Matrix Factorization vs SVD

# SVD (Singular Value Decomposition) yöntemi bir boyut indirgeme yöntemidir.

# ALS

#############################
# Adım 1: Veri Setinin Hazırlanması
#############################

# pip install surprise
import pandas as pd
from surprise import Reader, SVD, Dataset, accuracy
from surprise.model_selection import GridSearchCV, train_test_split, cross_validate
pd.set_option('display.max_columns', None)


movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
df = movie.merge(rating, how="left", on="movieId")
df.head()


movie_ids = [130219, 356, 4422, 541]
movies = ["The Dark Knight (2011)",
          "Cries and Whispers (Viskningar och rop) (1972)",
          "Forrest Gump (1994)",
          "Blade Runner (1982)"]

sample_df = df[df.movieId.isin(movie_ids)]
sample_df.shape
sample_df.head()


user_movie_df = sample_df.pivot_table(index=["userId"], columns=["title"], values="rating")
user_movie_df.head()

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(sample_df[['userId', 'movieId', 'rating']], reader)
type(data)

##############################
# Adım 2: Modelleme
##############################


trainset, testset = train_test_split(data, test_size=.25)

svd_model = SVD()
svd_model.fit(trainset)

predictions = svd_model.test(testset)

predictions

accuracy.rmse(predictions)

cross_validate(svd_model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
user_movie_df.head()


# userid 1 için blade runner tahmini yapalım: 541
svd_model.predict(uid=1.0, iid=541, verbose=True)

# userid 1 için Whispers tahmini yapalım: 356
svd_model.predict(uid=1.0, iid=356, verbose=True)



##############################
# Adım 3: Model Tuning
##############################


param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005]}

gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3, n_jobs=-1, joblib_verbose=True)

gs.fit(data)

gs.best_score['rmse']


gs.best_params['rmse']



##############################
# Adım 4: Final Model ve Tahmin
##############################


svd_model = SVD(**gs.best_params['rmse'])

data = data.build_full_trainset()
svd_model.fit(data)

user_movie_df.head()

# userid 1 için blade runner tahmini yapalım: 541
svd_model.predict(uid=1.0, iid=541, verbose=True)

svd_model.predict(uid=1.0, iid=356, verbose=True)


# https://en.wikipedia.org/wiki/Matrix_factorization_(recommender_systems)
# http://sifter.org/~simon/journal/20061211.html
# https://www.freecodecamp.org/news/singular-value-decomposition-vs-matrix-factorization-in-recommender-systems-b1e99bc73599/
# https://surprise.readthedocs.io/en/stable/getting_started.html