############################################
# SORTING PRODUCTS
############################################

# Rating Products
# - Average
# - Time-Based Weighted Average
# - User-Based Weighted Average
# - Weighted Rating
# - Uygulama: Kullanıcı ve Zaman Ağırlıklı Kurs Puanı Hesaplama


# Sorting Products
# - Sorting by Rating
# - Sorting by Comment Count or Purchase Count
# - Sorting by Rating, Comment and Purchase
# - Sorting by Bayesian Average Rating Score (Sorting Products with 5 Star Rated)
# - Hybrid Sorting: BAR Score + Diğer Faktorler
# - Uygulama: Kurs Sıralama
# - Uygulama: IMDB Movie Scoring & Sorting

# Sorting Reviews
# - Score Up-Down Diff
# - Average rating
# - Wilson Lower Bound Score
# - Uygulama: E-Ticaret Ürün Yorumlarının Sıralanması


###################################################
# Rating Products
###################################################

############################################
# Uygulama: Kullanıcı ve Zaman Ağırlıklı Kurs Puanı Hesaplama
############################################

import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


# (50+ Saat) Python A-Z™: Veri Bilimi ve Machine Learning
# Puan: 4.8 (4.764925)
# Toplam Puan: 4611
# Puan Yüzdeleri: 75, 20, 4, 1, <1
# Yaklaşık Sayısal Karşılıkları: 3458, 922, 184, 46, 6


df = pd.read_csv("5.Hafta/Ders Öncesi Notlar/course_reviews.csv")
df.head()

df["Rating"].value_counts()

df["Questions Asked"].value_counts()


df.groupby("Questions Asked").agg({"Questions Asked": "count",
                                   "Rating": "mean"})

df.head()

####################
# Average
####################

df["Rating"].mean()



####################
# Time-Based Weighted Average
####################

df['Timestamp'] = pd.to_datetime(df['Timestamp'])

current_date = pd.to_datetime('2021-02-10 0:0:0')

df["days"] = (current_date - df['Timestamp']).dt.days

df.head()


df.loc[df["days"] <= 30, "Rating"].mean() * 28 / 100 + \
df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean() * 26 / 100 + \
df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean() * 24 / 100 + \
df.loc[(df["days"] > 180), "Rating"].mean() * 22 / 100

def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[df["days"] <= 30, "Rating"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["days"] > 30) & (dataframe["days"] <= 90), "Rating"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["days"] > 90) & (dataframe["days"] <= 180), "Rating"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["days"] > 180), "Rating"].mean() * w4 / 100

time_based_weighted_average(df)


####################
# User-Based Weighted Average
####################

df.loc[df["Progress"] <= 10, "Rating"].mean() * 22 / 100 + \
df.loc[(df["Progress"] > 10) & (df["Progress"] <= 45), "Rating"].mean() * 24 / 100 + \
df.loc[(df["Progress"] > 45) & (df["Progress"] <= 75), "Rating"].mean() * 26 / 100 + \
df.loc[(df["Progress"] > 75), "Rating"].mean() * 28 / 100

def user_based_weighted_average(dataframe, w1=22, w2=24, w3=26, w4=28):
    return dataframe.loc[dataframe["Progress"] <= 10, "Rating"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["Progress"] > 10) & (dataframe["Progress"] <= 45), "Rating"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["Progress"] > 45) & (dataframe["Progress"] <= 75), "Rating"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["Progress"] > 75), "Rating"].mean() * w4 / 100


user_based_weighted_average(df)


####################
# Weighted Rating
####################

def course_weighted_rating(dataframe, time_w=50, user_w=50):
    return time_based_weighted_average(dataframe) * time_w / 100 + user_based_weighted_average(dataframe) * user_w / 100

course_weighted_rating(df)

course_weighted_rating(df, time_w=40, user_w=60)


###################################################
# Sorting Products
###################################################

###################################################
# Uygulama: Kurs Sıralama
###################################################

df = pd.read_csv("5.Hafta/Ders Öncesi Notlar/product_sorting.csv")
df.head()

####################
# Sorting by Rating
####################

df.sort_values("rating", ascending=False).head(20)

####################
# Sorting by Comment Count or Purchase Count
####################

df.sort_values("purchase_count", ascending=False).head(20)

df.sort_values("commment_count", ascending=False).head(20)

####################
# Sorting by Rating, Comment and Purchase
####################

df["purchase_count_scaled"] = MinMaxScaler(feature_range=(1, 5)). \
    fit(df[["purchase_count"]]). \
    transform(df[["purchase_count"]])

df["commment_count_scaled"] = MinMaxScaler(feature_range=(1, 5)). \
    fit(df[["commment_count"]]). \
    transform(df[["commment_count"]])

df.head()

(df["commment_count_scaled"] * 32 / 100 +
 df["purchase_count_scaled"] * 26 / 100 +
 df["rating"] * 42 / 100)

def weighted_sorting_score(dataframe, w1=32, w2=26, w3=42):
    return (dataframe["commment_count_scaled"] * w1 / 100 +
            dataframe["purchase_count_scaled"] * w2 / 100 +
            dataframe["rating"] * w3 / 100)


df["weighted_sorting_score"] = weighted_sorting_score(df)

df.sort_values("weighted_sorting_score", ascending=False).head(20)

# df[df["course_name"].str.contains("Veri Bilimi")].sort_values("weighted_sorting_score", ascending=False).head(20)


####################
# Bayesian Average Rating Score
####################

def bayesian_average_rating(n, confidence=0.95):
    if sum(n) == 0:
        return 0
    K = len(n)
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    for k, n_k in enumerate(n):
        first_part += (k + 1) * (n[k] + 1) / (N + K)
        second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
    return score


def bayesian_average_rating(n, confidence=0.95):
    """
    Olasılıksal

    Parameters
    ----------
    n: list or df
        puanların frekanslarını tutar.
        Örnek: [2, 40, 56, 12, 90] 2 tane 1 puan, 40 tane 2 puan, ... , 90 tane 5 puan.
    confidence: float
        güven aralığı

    Returns
    -------
    BAR score: float


    """

    # rating'lerin toplamı sıfır ise sıfır dön.
    if sum(n) == 0:
        return 0
    # eşsiz yıldız sayısı. 5 yıldızdan da puan varsa 5 olacaktır.
    K = len(n)
    # 0.95'e göre z skoru.
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    # toplam rating sayısı.
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    # index bilgisi ile birlikte yıldız sayılarını gez.
    # formülasyondaki hesapları gerçekleştir.
    for k, n_k in enumerate(n):
        first_part += (k + 1) * (n[k] + 1) / (N + K)
        second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
    return score


df["bar_sorting_score"] = df.apply(lambda x: bayesian_average_rating(x[["1_point",
                                                                        "2_point",
                                                                        "3_point",
                                                                        "4_point",
                                                                        "5_point"]]), axis=1)

df.sort_values("weighted_sorting_score", ascending=False).head(20)

df.sort_values("bar_sorting_score", ascending=False).head(20)


####################
# Hybrid Sorting: BAR Score + Diğer Faktorler
####################


def hybrid_sorting_score(dataframe, bar_w=60, wss_w=40):
    bar_score = dataframe.apply(lambda x: bayesian_average_rating(x[["1_point",
                                                                     "2_point",
                                                                     "3_point",
                                                                     "4_point",
                                                                     "5_point"]]), axis=1)
    wss_score = weighted_sorting_score(dataframe)

    return bar_score*bar_w/100 + wss_score*wss_w/100


df["hybrid_sorting_score"] = hybrid_sorting_score(df)

df.sort_values("hybrid_sorting_score", ascending=False).head(20)

# df[df["course_name"].str.contains("Veri Bilimi")].sort_values("hybrid_sorting_score", ascending=False).head(20)

############################################
# Uygulama: IMDB Movie Scoring & Sorting
############################################


import pandas as pd
import math
import scipy.stats as st

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv("datasets/the_movies_dataset/movies_metadata.csv",
                 low_memory=False)  # DtypeWarning kapamak icin
df.head()

df = df[["title", "vote_average", "vote_count"]]


########################
# Vote Average'a Göre Sıralama
########################

df.sort_values("vote_average", ascending=False).head(20)

df["vote_count"].describe([0.10, 0.25, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99]).T


df[df["vote_count"] > 400].sort_values("vote_average", ascending=False).head(20)


########################
# vote_count
########################

df[df["vote_count"] > 400].sort_values("vote_count", ascending=False).head(20)


########################
# vote_average * vote_count
########################

df["average_count_score"] = df["vote_average"] * df["vote_count"]

df.sort_values("average_count_score", ascending=False).head(20)




########################
# weighted_rating
########################

# weighted_rating = (v/(v+M) * r) + (M/(v+M) * C)


# r = vote average
# v = vote count
# M = minimum votes required to be listed in the Top 250
# C = the mean vote across the whole report (currently 7.0)


# Film 1:
# r = 8
# M = 500
# v = 1000

# (1000 / (1000+500))*8 = 5.33


# Film 2:
# r = 8
# M = 500
# v = 3000

# (3000 / (3000+500))*8 = 6.85



# Film 1:
# r = 8
# M = 500
# v = 1000

# Birinci bölüm:
# (1000 / (1000+500))*8 = 5.33

# İkinci bölüm:
# 500/(1000+500) * 7 = 2.33

# Toplam = 5.33 + 2.33 = 7.66

# Film 2:
# r = 8
# M = 500
# v = 3000

# Birinci bölüm:
# (3000 / (3000+500))*8 = 6.85

# İkinci bölüm:
# 500/(3000+500) * 7 = 1

# Toplam = 7.85



M = 2500
C = df['vote_average'].mean()

def weighted_rating(r, v, M, C):
    return (v / (v + M) * r) + (M / (v + M) * C)


df.sort_values("average_count_score", ascending=False).head(20)

weighted_rating(7.40000, 11444.00000, M, C)

weighted_rating(8.10000, 14075.00000, M, C)

weighted_rating(8.50000, 8358.00000, M, C)

df["weighted_rating"] = weighted_rating(df["vote_average"], df["vote_count"], M, C)

df.sort_values("weighted_rating", ascending=False).head(20)


####################
# Bayesian Average Rating Score
####################

def bayesian_average_rating(n, confidence=0.95):
    if sum(n) == 0:
        return 0
    K = len(n)
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    for k, n_k in enumerate(n):
        first_part += (k + 1) * (n[k] + 1) / (N + K)
        second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
    return score



# esaretin bedeli (9,2)
bayesian_average_rating([34733, 4355, 4704, 6561, 13515, 26183, 87368, 273082, 600260, 1295351])

# baba (9,1)
bayesian_average_rating([37128, 5879, 6268, 8419, 16603, 30016, 78538, 199430, 402518, 837905])

# baba2 (9)
bayesian_average_rating([20469, 3892, 4347, 6210, 12657, 26349, 70845, 175492, 324898, 486342])

# karasovalye (9)
bayesian_average_rating([30345, 7172, 8083, 11429, 23236, 49482, 137745, 354608, 649114, 1034843])

# deadpole
bayesian_average_rating([10929, 4248, 5888, 9817, 21897, 59973, 153250, 256674, 197525, 183404])


df = pd.read_csv("datasets/imdb_ratings.csv")
df = df.iloc[0:, 1:]
df.head()


df["bar_score"] = df.apply(lambda x: bayesian_average_rating(x[["one", "two", "three", "four", "five",
                                                                "six", "seven", "eight", "nine", "ten"]]), axis=1)

df.sort_values("bar_score", ascending=False).head(20)

############################################
# SORTING REVIEWS
############################################
import pandas as pd
import math
import scipy.stats as st
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

###################################################
# Up-Down Diff Score = (up ratings) − (down ratings)
###################################################

def score_up_down_diff(up, down):
    return up - down


# Review 1: 600 up 400 down total 1000
# Review 2: 5500 up 4500 down total 10000

# Review 1 Score:
score_up_down_diff(600, 400)

# Review 2 Score
score_up_down_diff(5500, 4500)

# Review 1 up yüzdesi nedir? Yüzde 60
# Review 2 up yüzdesi nedir? Yüzde 55

###################################################
# Score = Average rating = (up ratings) / (total ratings)
###################################################


def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

score_average_rating(600, 400)
score_average_rating(5500, 4500)


# Review 1: 2 up 0 down total 2
# Review 2: 100 up 1 down total 101

# Review 1 Average Rating Score: 2/2 = 1
# Review 2 Average Rating Score: 100/101 = 0.99


###################################################
# Wilson Lower Bound Score
###################################################

# p = ilgilenen olay / tüm olay
# p için güven aralığı



def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.

    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

wilson_lower_bound(600, 400)
wilson_lower_bound(5500, 4500)

wilson_lower_bound(2, 0)
wilson_lower_bound(100, 1)


###################################################
# Case Study:
###################################################

up = [1115, 454, 258, 253, 220, 227, 127, 75, 60, 67, 38, 11, 26, 44, 1, 0, 6, 15, 20]
down = [43, 35, 26, 19, 9, 16, 8, 8, 4, 9, 1, 0, 0, 5, 0, 0, 0, 0, 3]
comments = pd.DataFrame({"up": up, "down": down})

comments["score_up_down_diff"] = comments.apply(lambda x: score_up_down_diff(x["up"],
                                                                             x["down"]), axis=1)

comments["score_average_rating"] = comments.apply(lambda x: score_average_rating(x["up"],
                                                                                 x["down"]),
                                                  axis=1)



comments["wilson_lower_bound"] = comments.apply(lambda x: wilson_lower_bound(x["up"],
                                                                             x["down"]),
                                                axis=1)

comments.sort_values("wilson_lower_bound", ascending=False)



###################################################
# Case Study
###################################################

up = [15, 70, 14, 4, 2, 5, 8, 37, 21, 52, 28, 147, 61, 30, 23, 40, 37, 61, 54, 18, 12, 68]
down = [0, 2, 2, 2, 15, 2, 6, 5, 23, 8, 12, 2, 1, 1, 5, 1, 2, 6, 2, 0, 2, 2]
comments = pd.DataFrame({"up": up, "down": down})


# score_pos_neg_diff
comments["score_pos_neg_diff"] = comments.apply(lambda x: score_up_down_diff(x["up"], x["down"]), axis=1)

# score_average_rating
comments["score_average_rating"] = comments.apply(lambda x: score_average_rating(x["up"], x["down"]), axis=1)

# wilson_lower_bound
comments["wilson_lower_bound"] = comments.apply(lambda x: wilson_lower_bound(x["up"], x["down"]), axis=1)


comments.sort_values("wilson_lower_bound", ascending=False)



# 1 ürün gösterildiği gibi değil. kandırılmayın.
# 2 ürün gösterildiği gibi değil. kandırılmayın.
# 3 ürün gösterildiği gibi değil. kandırılmayın.