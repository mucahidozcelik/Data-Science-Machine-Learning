###############################################
# PYTHON İLE VERİ ANALİZİ
###############################################

# - NumPy
# - Pandas
# - Veri Görselleştirme: Seaborn & Matplotlib
# - Gelişmiş Fonksiyonel Keşifçi Veri Analizi (Advanced Functional EDA)

#############################################
# NUMPY
#############################################

# Neden NumPy? (Why NumPy?)
# NumPy Array'i Oluşturmak (Creating Numpy Arrays)
# NumPy Array Özellikleri (Attibutes of Numpy Arrays)
# Yeniden Şekillendirme (Reshaping)
# Index Seçimi (Index Selection)
# Slicing
# Fancy Index
# Numpy'da Koşullu İşlemler (Conditions on Numpy)
# Matematiksel İşlemler (Mathematical Operations)

# Verimli veri saklama ve vektörel operasyonlardır.

#############################################
# Neden NumPy?
#############################################

#######################
# Python Lists vs. Numpy Arrays
#######################

# listeler elemanlarının her birisi için tip , boyut vs bilgilerini tutar
# ndarray ise sadece tek bir bilgi tutar ( hepsi int , float gibi )
# numpy array i listelerden daha hızlıdır


import numpy as np

print(np.__version__)

import time

size_of_vec = 10000


def pure_python_version():
    t1 = time.time()
    x = range(size_of_vec)
    y = range(size_of_vec)
    z = [x[i] + y[i] for i in range(len(x))]
    return time.time() - t1


def numpy_version():
    t1 = time.time()
    x = np.arange(size_of_vec)
    y = np.arange(size_of_vec)
    z = x + y
    return time.time() - t1


t1 = pure_python_version()
t2 = numpy_version()

print(t1, t2)
print("Numpy is in this example " + str(t1 / t2) + " faster!")

# source:
# https://webcourses.ucf.edu/courses/1249560/pages/python-lists-vs-numpy-arrays-what-is-the-difference


#######################
# Low Level vs High Level
#######################

# normalde bir listeye herhangi bir işlem yapabilmek için bütün elemanlarını tek tek yazarak
# işlem yapıyoruz ama array le bu işlemleri yaparken elemanları tek tek yazmaya gerek kalmıyor
# array tek işlemle bütün elemanlarını dolaşıyor
# numpy ın önemli olan kısmı burasıdır


a = [1, 2, 3, 4]
b = [2, 3, 4, 5]
ab = []

for i in range(0, len(a)):
    ab.append(a[i] * b[i])

ab


a = np.array([1, 2, 3, 4])
b = np.array([2, 3, 4, 5])
a * b


#############################################
# NumPy Array'i Oluşturmak (Creating Numpy Arrays)
#############################################

import numpy as np

np.array([1, 2, 3, 4, 5])

type(np.array([1, 2, 3, 4, 5]))

np.zeros(10, dtype=int)

np.random.randint(0, 10, size=10)

np.random.normal(10, 4, (3, 4))

np.random.randint(0, 10, (3, 3))


#############################################
# NumPy Array Özellikleri (Attibutes of Numpy Arrays)
#############################################

# ndim: boyut sayısı
# shape: boyut bilgisi
# size: toplam eleman sayısı
# dtype: array veri tipi

a = np.random.randint(10, size=10)


a.ndim
a.shape
a.size
a.dtype


#############################################
# Yeniden Şekillendirme (Reshaping)
#############################################

np.arange(1, 10)
np.arange(1, 10).reshape((3, 3))


#############################################
# Index Seçimi (Index Selection)
#############################################

a = np.random.randint(10, size=10)

a[0]
a[-1]
a[0] = 999
a[0] = 1.2

m = np.random.randint(10, size=(3, 5))

m[0, 0]
m[1, 1]
m[2, 3]
m[2, 3] = 9999

m[2, 3] = 2.9


m[:, 0]
m[1, :]

m[0:2, 0:3]




#############################################
# Slicing
#############################################

a = np.random.randint(10, size=30)
a[1:10]

m = np.random.randint(10, size=(3, 5))
m[:, 0]
m[1, :]
m[0:2, 0:3]

#############################################
# Step
#############################################

step = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
step[1:10:2]


#############################################
# Fancy Index
#############################################

v = np.arange(0, 30, 3)

v[1]
v[4]
v[3]

catch = [1, 2, 3]

v[catch]

m = np.arange(9).reshape((3, 3))

m[2, [1, 2]]

m[0:1, [1, 2]]


#############################################
# Numpy'da Koşullu İşlemler (Conditions on Numpy)
#############################################

v = np.array([1, 2, 3, 4, 5])
v

ab = []

for i in v:
    if i < 3:
        ab.append(i)

ab

#######################
# Numpy ile
#######################

# Önemli


v[v < 3]
v[v >= 3]
v[v <= 3]
v[v == 3]


#############################################
# Matematiksel İşlemler (Mathematical Operations)
#############################################

v = np.array([1, 2, 3, 4, 5])
v / 5

v * 5 / 10
v ** 2
v - 1

np.subtract(v, 1)
np.add(v, 1)
np.mean(v)
np.sum(v)
np.min(v)
np.max(v)
np.var(v)


#######################
# NumPy ile İki Bilinmeyenli Denklem Çözümü
#######################

# 5*x0 + x1 = 12
# x0 + 3*x1 = 10

a = np.array([[5, 1], [1, 3]])
b = np.array([12, 10])

np.linalg.solve(a, b)


#############################################
# PANDAS
#############################################

# Pandas Series
# Veri Okuma (Reading Data)
# Veriye Hızlı Bakış (Quick Look at Data)
# Pandas'ta Seçim İşlemleri (Selection in Pandas)
# Toplulaştırma ve Gruplama (Aggregation & Grouping)
# Apply ve Lambda
# Birleştirme (Join) İşlemleri


#############################################
# Pandas Series
#############################################


import pandas as pd
s = pd.Series([10, 88, 3, 4, 5])

type(s)


s.index
s.dtype
s.size
s.ndim
s.values
s.head(3)
s.tail(3)

#############################################
# Veri Okuma (Reading Data)
#############################################

df = pd.read_csv('2.Hafta/Ders Öncesi Notlar/titanic.csv')

#############################################
# Bir Dizindeki Birden Fazla CSV Dosyasını Okumak ve Tek Bir DF'e Çevirmek
#############################################

import glob


all_files = glob.glob(r'2.Hafta/Ders Öncesi Notlar' + "/*.csv")

file_list = [pd.read_csv(f) for f in all_files]

all_df = pd.concat(file_list, axis=0, ignore_index=True)

#alternatif
all_df = pd.concat(map(pd.read_csv,
                       glob.glob('2.Hafta/Ders Öncesi Notlar/*.csv')))




#############################################
# Veriye Hızlı Bakış (Quick Look at Data)
#############################################

df = pd.read_csv('2.Hafta/Ders Öncesi Notlar/titanic.csv')

df.head()

df.tail()

df.shape

df.info()

df.columns

df.index

df.describe().T

df.isnull().values.any()

df.isnull().sum()

df["Sex"].value_counts()

df["Sex"]


#############################################
# Pandas'ta Seçim İşlemleri (Selection in Pandas)
#############################################

# - Index Üzerinde İşlemler
# - Değişkeni Indexe Çevirmek
# - Indexi Değişkene Çevirmek
# - Değişkenler Üzerinde İşlemler
# - Value'lar Üzerinde İşlemler
# - iloc & loc
# - Koşullu Seçim (Conditional Selection)

df.head()

type(df)


#######################
# Index Üzerinde İşlemler
#######################

#önemli !


df.columns
df.index


df[13:18]

df.drop(0, axis=0).head()

#birden fazla index silme
delete_indexes = [1, 3, 5, 7]

df.drop(delete_indexes, axis=0).head(10)

#kalıcı olarak silme (inplace)
df.drop(delete_indexes, axis=0, inplace=True)


#######################
# Değişkeni Indexe Çevirmek
#######################

# Önemli !

df = pd.read_csv('2.Hafta/Ders Öncesi Notlar/titanic.csv')

df.sort_values("PassengerId").head()

df = df.sort_values("PassengerId")

df.index = df["PassengerId"]

df.drop("PassengerId", axis=1).head()

df.loc[:, df.columns != 'PassengerId'].head()

df.drop("PassengerId", axis=1, inplace=True)




#######################
# Indexi Değişkene Çevirmek
#######################

df.index
df["PassengerId"] = df.index


df.drop("PassengerId", axis=1, inplace=True)

#önemli
df.head()
df.reset_index().head()

df = df.reset_index()

#######################
# Değişkenler Üzerinde İşlemler
#######################

#bir değişkenin varlığını sorgulama
"Age" in df

#bir değişkeni seçme
df["Age"].head()


type(df["Age"])
type(df)

#pandas series i dataframe e çevirme
df[["Age"]].head()

type(df[["Age"]])

#farklı bir şekilde değişken seçme
df.Age.head()

#birden fazla değişken seçme
df[["Age", "Pclass"]].head()

#belirli bir listeye göre değişken seçme
col_names = ["Age", "Embarked", "Ticket"]

df[col_names].head()

#veri setine yeni değişken ekleme
df["Age2"] = df["Age"] ** 2
df["Age3"] = df["Age"] / df["Age2"]

df.head()

#değişken silme
df.drop("Age3", axis=1).head()

#bir listeye göre değişken silme
col_names = ["Age", "Embarked", "Ticket"]

df.drop(col_names, axis=1).head()

df.head()

df.loc[:, ~df.columns.str.contains('Age')].head()

df.loc[:, df.columns.str.contains('Age')].head()


#######################
# Value'lar Üzerinde İşlemler
#######################

df.values

for row in df.values:
    print(row)


#######################
# iloc & loc
#######################

# iloc: integer based selection

df.head()
df.iloc[0:3]
df.iloc[0, 0]


# loc: label based selection
df.loc[0:3]

df[0:3]

#klasik seçme yöntemiyle df[] değişken seçemiyoruz isteklerimizi karşılamıyor
# o yüzden loc&iloc yapısı kullanıyoruz.
# özellikle değişkenler için seçim yapmak istediğimizde loc yapısı kullanılır .

df[0:3, "Age"]

df.iloc[0:3, "Age"]

df.loc[0:3, "Age"]

col_names = ["Age", "Embarked", "Ticket"]

df.loc[0:3, col_names]


#######################
# Koşullu Seçim (Conditional Selection)
#######################

df["Age"].head()
df.Age.head()

df[["Age", "Fare", "Cabin"]].head()

df["Age"] > 50

df[df["Age"] > 50]

df[df["Age"] > 50].count()

df[df["Age"] > 50]["PassengerId"].count()

df[df["Age"] > 50]["Name"]

df[df["Age"] > 50]["Name"].nunique()

df.loc[df["Age"] > 50, "Name"].head()

df[(df["Age"] > 50) & (df["Sex"] == "female")].head()

df[(df["Age"] > 50) & (df["Sex"] == "female")]["PassengerId"].count()

df[df["Age"] > 50]["PassengerId"].count()


#############################################
# Toplulaştırma ve Gruplama (Aggregation & Grouping)
#############################################

# - count()
# - first()
# - last()
# - mean()
# - median()
# - min()
# - max()
# - std()
# - var()
# - sum()
# - pivot table



df[df["Age"] > 50][["PassengerId", "Sex"]].groupby("Sex").agg({"count"})


df.loc[df["Age"] > 50, ["PassengerId", "Sex"]].groupby("Sex").agg("count")

df.loc[df["Age"] > 50, ["Age", "Sex"]].groupby("Sex").agg(["count", "mean"])


df.loc[df["Age"] > 50, ["PassengerId", "Age", "Sex"]].groupby("Sex").agg({"PassengerId": "count",
                                                                          "Age": ["min", "max", "mean"]})

df.loc[df["Age"] > 50].groupby(["Sex", "Embarked", "Pclass"]).agg({"PassengerId": "count",
                                                                   "Age": ["min", "max", "mean"]})


df.groupby(["Sex", "Embarked", "Pclass"]).agg({"PassengerId": "count",
                                               "Age": ["min", "max", "mean"]})


agg_functions = ["nunique", "first", "last", "sum", "var", "std"]

df.groupby(["Sex", "Embarked", "Pclass"]).agg({"PassengerId": "count",
                                               "Age": agg_functions})


#######################
# Pivot table
#######################

df = pd.read_csv('2.Hafta/Ders Öncesi Notlar/titanic.csv')


def load_titanic():
    return pd.read_csv('2.Hafta/Ders Öncesi Notlar/titanic.csv')

df = load_titanic()

df.pivot_table(values="Age" , index="Sex" , columns="Embarked")

df.pivot_table(values="Survived" , index="Sex" , columns="Embarked")

df.pivot_table(values="Age", index="Sex", columns="Embarked", aggfunc="mean")

df.pivot_table(values="Age", index="Sex", columns="Embarked", aggfunc="std")

#######################
# Sayısal Değişkenin Kategorik Değişkene Çevrilmesi
#######################

df.head()

df["new_age"] = pd.cut(df["Age"], [0, 10, 18, 25, 40, 90])

#argümanların sıralamasını biliyorsan yazmana gerek yok (values,index,columns)

df.pivot_table("Survived", index="Sex", columns="new_age")

df.pivot_table("Survived", index=["Sex", "Pclass"], columns="new_age")


#############################################
# Apply ve Lambda
#############################################

(df["Age"] ** 2).head()
(df["Age2"] ** 2).head()
(df["Age3"] ** 2).head()


for col in df.columns:
    if "Age" in col:
        print(col)

for col in df.columns:
    if "Age" in col:
        print((df[col] ** 2).head())

for col in df.columns:
    if "Age" in col:
        df[col] = df[col] ** 2

# lambda , atama kaydetme işlemi yapmadan fonksiyon tanımlama yoludur
# apply fonksiyonu sütunlara uygular
# apply default olarak axis=0 yani sütunlarda işlem yapıyor

df[["Age", "Age2", "Age3"]].apply(lambda x: x ** 2).head()

df.loc[:, df.columns.str.contains('Age')].apply(lambda x: x ** 2).head()

df.loc[:, df.columns.str.contains('Age')].apply(lambda x: (x - x.mean()) / x.std()).head()

df[["Age"]].apply(lambda x: (x - x.mean()) / x.std()).head()

def standart_scaler(col_name):
    return (col_name - col_name.mean()) / col_name.std()

standart_scaler(df["Age"]).head()

df[["Age"]].apply(standart_scaler).head()

df.loc[:, df.columns.str.contains('Age')].apply(standart_scaler).head()

df.loc[:, df.columns.str.contains('Age')].apply(lambda x: (x - x.mean()) / x.std()).head()

df.loc[:, ["Age", "Age2", "Age3"]] = df.loc[:, df.columns.str.contains('Age')].apply(standart_scaler)


#############################################
# Birleştirme (Join) İşlemleri
#############################################

import numpy as np
import pandas as pd

m = np.random.randint(1, 30, size=(5, 3))
df1 = pd.DataFrame(m, columns=["var1", "var2", "var3"])
df1

df2 = df1 + 99

df2

#######################
# concat ile Birleştirme İşlemleri
#######################

# alt alta birleştirmek
pd.concat([df1, df2])

# indexlerde problem var gibi gözüküyor
pd.concat([df1, df2], ignore_index=True)

# Peki değişken isimleri farklı olsaydı?
df2.columns = ["var1", "var2", "deg3"]
df1.columns

# Birleşen df'te boşluklar olacak.
pd.concat([df1, df2])

# Eğer sadece kesişen isimler ile birleştirme yapmak istersek:
pd.concat([df1, df2], join="inner")

#######################
# Merge ile Birleştirme İşlemleri
#######################

df1 = pd.DataFrame({'employees': ['john', 'dennis', 'mark', 'maria'],
                    'group': ['accounting', 'engineering', 'engineering', 'hr']})

df2 = pd.DataFrame({'employees': ['mark', 'john', 'dennis', 'maria'],
                    'start_date': [2010, 2009, 2014, 2019]})

# Amaç: Her çalışanın işe başlangıç tarihine ulaşmak
pd.merge(df1, df2)
pd.merge(df1, df2, on="employees")

# Amaç: Her çalışanın müdürünün bilgisine erişmek
df3 = pd.merge(df1, df2)

df4 = pd.DataFrame({'group': ['accounting', 'engineering', 'hr'],
                    'manager': ['Caner', 'Mustafa', 'Berkcan']})

pd.merge(df3, df4)

# Amaç: Meslek grubu yeteneklerinin kişilerle eşleştirilmesi
df5 = pd.DataFrame({'group': ['accounting', 'accounting', 'engineering', 'engineering', 'hr', 'hr'],
                    'skills': ['math', 'excel', 'coding', 'linux', 'excel', 'management']})

pd.merge(df1, df5)



#############################################
# VERİ GÖRSELLEŞTİRME: SEABORN & MATPLOTLIB
#############################################

#############################################
# MATPLOTLIB
#############################################

import matplotlib.pyplot as plt


#############################################
# Kategorik Değişken Görselleştirme
#############################################
import numpy as np
import pandas as pd

def load_titanic():
    return pd.read_csv('2.Hafta/Ders Öncesi Notlar/titanic.csv')

df = load_titanic()

df['Sex'].value_counts().plot(kind='bar', rot=0)
plt.show()


#############################################
# Sayısal Değişken Görselleştirme
#############################################
import seaborn as sns
df = sns.load_dataset("tips")

# Histogram
plt.hist(df["total_bill"])
plt.show()

# Boxplot
plt.boxplot(df["total_bill"])
plt.show()



#############################################
# Matplotlib'in Özellikleri
#############################################

#######################
# plot
#######################

# İki nokta arasında çizgi çizmek
x = np.array([1, 8])
y = np.array([0, 150])

plt.plot(x, y)
plt.show()

# Çizgisiz sadece noktaları göstermek
plt.plot(x, y, 'o')
plt.show()

# Daha fazla sayıda nokta
x = np.array([2, 4, 6, 8, 10])
y = np.array([1, 3, 5, 7, 9])

plt.plot(x, y)
plt.show()

#######################
# marker
#######################

y = np.array([13, 28, 11, 100])

# y noktalarına içi dolu daire koymak
plt.plot(y, marker='o')
plt.show()

# y noktalarına yıldız koymak
plt.plot(y, marker='*')
plt.show()

# bu markerlerdan birçok var.
markers = ['o', '*', '.', ',', 'x', 'X', '+', 'P', 's', 'D', 'd', 'p', 'H', 'h']

for marker in markers:
    plt.plot(y, marker)
    plt.show()

#######################
# line
#######################

y = np.array([13, 28, 11, 100])
plt.plot(y)
plt.show()

# birçok çizgi çeşidi de var
styles = ['dotted', 'dashed', 'dashdot']
for style in styles:
    plt.plot(y, linestyle=style)
    plt.show()

# e birçok renk de var
# hem rengi hem de stilleri değiştirerek yazdıralım
# ve hatta patronu çıldırtalım.
styles = ['dotted', 'dashed', 'dashdot']
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']

for style in styles:
    for color in colors:
        plt.plot(y, linestyle=style, color=color)
        plt.title(style + " " + color)
        plt.show()

# şimdi daha da çıldırtalım patronu
# ihtiyacımız şu olsun.
# biz sözlük var.
# key'lerinde arguman değeri olabilecek stringler value'larında da bu argumanların tanımları olsun.

colors_dict = {'r': "red", 'g': "green", 'b': "blue", 'c': "cyan",
               'm': "magenta", 'y': "yellow", 'k': "black", 'w': "white"}

for style in styles:
    for color in colors_dict.keys():
        plt.plot(y, linestyle=style, color=color)
        plt.title(style.upper() + " " + colors_dict[color])
        plt.show()

#######################
# Multiple Lines
#######################

x = np.array([23, 18, 31, 10])
y = np.array([13, 28, 11, 100])
plt.plot(x)
plt.plot(y)
plt.show()

#######################
# Labels
#######################

x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.plot(x, y)

# Başlık
plt.title("Bu ana başlık")
# X eksenini isimlendirme
plt.xlabel("X ekseni isimlendirmesi")
# Y eksenini isimlendirme
plt.ylabel("Y ekseni isimlendirmesi")
# Izgara
plt.grid()

plt.show()

#######################
# Subplots
#######################


# 2 farklı grafiği tek satır iki sütun olarak konumlamak.
# plot 1
x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.subplot(1, 2, 1)
plt.title("1")
plt.plot(x, y)

# plot 2
x = np.array([8, 8, 9, 9, 10, 15, 11, 15, 12, 15])
y = np.array([24, 20, 26, 27, 280, 29, 30, 30, 30, 30])
plt.subplot(1, 2, 2)
plt.title("2")
plt.plot(x, y)

plt.show()

# 3 grafiği bir satır 3 sütun olarak konumlamak.
# plot 1
x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.subplot(1, 3, 1)
plt.title("1")
plt.plot(x, y)

# plot 2
x = np.array([8, 8, 9, 9, 10, 15, 11, 15, 12, 15])
y = np.array([24, 20, 26, 27, 280, 29, 30, 30, 30, 30])
plt.subplot(1, 3, 2)
plt.title("2")
plt.plot(x, y)

# plot 3
x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.subplot(1, 3, 3)
plt.title("3")
plt.plot(x, y)

plt.show()

#############################################
# SEABORN
#############################################

# Kategorik Değişkenler: countplot
# Sayısal Değişkenler: histogram ve boxplot

import seaborn as sns
from matplotlib import pyplot as plt

df = sns.load_dataset("tips")

df.describe().T


#############################################
# Kategorik Değişken Görselleştirme
#############################################

df["sex"].value_counts()

sns.countplot(x=df["sex"], data=df)
plt.show()




#############################################
# Sayısal Değişken Görselleştirme
#############################################

# pandas ile histogram
df["total_bill"].hist()
plt.show()

# seaborn ile boxplot
sns.boxplot(x=df["total_bill"])
plt.show()


#############################################
# GELİŞMİŞ FONKSİYONEL KEŞİFÇİ VERİ ANALİZİ (ADVANCED FUNCTIONAL EDA)
#############################################

# 1. Genel Resim
# 2. Kategorik Değişken Analizi (Analysis of Categorical Variables)
# 3. Sayısal Değişken Analizi (Analysis of Numerical Variables)
# 4. Hedef Değişken Analizi (Analysis of Target Variable)
# 5. Korelasyon Analizi (Analysis of Correlation)


#############################################
# 1. Genel Resim
#############################################

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

pd.set_option('display.max_columns', None)

def load_titanic():
    df = pd.read_csv("2.Hafta/Ders Öncesi Notlar/titanic.csv")
    return df


def load_app_train():
    df = pd.read_csv("2.Hafta/Ders Öncesi Notlar/application_train.csv")
    return df

df = load_titanic()



df.head()
df.tail()
df.shape
df.info()
df.columns
df.index
df.describe().T
df.isnull().values.any()
df.isnull().sum()


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)

    print("##################### Types #####################")
    print(dataframe.dtypes)

    print("##################### Head #####################")
    print(dataframe.head(head))

    print("##################### Tail #####################")
    print(dataframe.tail(head))

    print("##################### NA #####################")
    print(dataframe.isnull().sum())

    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

dff = load_app_train()

check_df(dff)


#############################################
# 2. Kategorik Değişken Analizi (Analysis of Categorical Variables)
#############################################

#kategorik değişken sınıfları ve frekanksları
df["Sex"].value_counts()

#kategorik değişkenin sınıfları
df["Sex"].unique()

#eşsiz sınıf sayıları
df["Sex"].nunique()

dff.head()

# kaç kategorik değişken var ve isimleri neler ?
cat_cols = [col for col in df.columns if df[col].dtypes == "O"]

df.head()

# numerik görünümlü kategorikleri bulmak için yazdık
num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes != "O"]

# kategorik ama kardinal değişkenleri bulmak için
cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and df[col].dtypes == "O"]

cat_cols
num_but_cat
cat_but_car

cat_cols = cat_cols + num_but_cat
2
cat_cols = [col for col in cat_cols if col not in cat_but_car]

df[cat_cols].nunique()

dff.shape

100 * df["Sex"].value_counts() / len(df)

def cat_summary(dataframe, col_name, plot=False):

    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

cat_summary(df, "Survived", plot=True)

for col in cat_cols:
    cat_summary(df, col, plot=True)


#############################################
# 3. Sayısal Değişken Analizi (Analysis of Numerical Variables)
#############################################

df.describe().T

df[["Age", "Fare"]].describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99]).T

num_cols = [col for col in df.columns if df[col].dtypes != 'O']

num_cols = [col for col in df.columns if df[col].dtypes != 'O' and col not in ["PassengerId", "Survived"]]

num_cols = [col for col in num_cols if col not in cat_cols]

cat_cols
num_cols

df["Age"].hist(bins=30)
plt.show()

sns.boxplot(x=df["Age"])
plt.show()


def num_summary(dataframe, numerical_col, plot=False):

    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]

    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()


num_summary(df, "Age")

for col in num_cols:
    num_summary(df, col, plot=True)


#############################################
# Kategorik ve Sayısal Değişken Analizinin Genelleştirilmesi
#############################################

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.

    """


    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat

    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]

    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(dff)


for col in cat_cols:
    cat_summary(dff, col , True)

for col in num_cols:
    num_summary(dff, col, plot=True)


import seaborn as sns
df = sns.load_dataset("iris")
print(grab_col_names(df))


#############################################
# 4. Hedef Değişken Analizi (Analysis of Target Variable)
#############################################

#######################
# Hedef Değişkenin Kategorik Değişkenler ile Analizi
#######################

df = load_titanic()

cat_cols, num_cols, cat_but_car = grab_col_names(df)

def target_summary_with_cat(dataframe, target, categorical_col):

    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


target_summary_with_cat(df, "Survived", "Sex")

for col in cat_cols:
    target_summary_with_cat(df, "Survived", col)


#######################
# Hedef Değişkenin Sayısal Değişkenler ile Analizi
#######################

# önemli bir konu

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

target_summary_with_num(df, "Survived", "Age")


for col in num_cols:
    target_summary_with_num(df, "Survived", col)


#############################################
# 5. Korelasyon Analizi (Analysis of Correlation)
#############################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("2.Hafta/Ders Öncesi Notlar/breast_cancer.csv")


df = df.iloc[:, 1:-1]

len(df.columns)

num_cols = [col for col in df.columns if df[col].dtype in [int, float]]

corr = df[num_cols].corr()

# korelasyon -1 +1 arasında değerler alır
# +1 e yakınsa pozitif yönlü ilişki -1e yakınsa negatif yönlü ilişki
# 0 ise ilişki yoktur denir .
# bir değişken artarken diğeri de artıyorsa + yönlü ilişki ,
# biri artarken diğeri azalıyorsa - yönlü ilişki olur

sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap="RdBu")
plt.show()


#######################
# Yüksek Korelasyonlu Değişkenlerin Silinmesi
#######################
import pandas as pd
pd.set_option("display.max_columns" , 5)

cor_matrix = df.corr().abs()

#           0         1         2         3
# 0  1.000000  0.117570  0.871754  0.817941
# 1  0.117570  1.000000  0.428440  0.366126
# 2  0.871754  0.428440  1.000000  0.962865
# 3  0.817941  0.366126  0.962865  1.000000


#     0        1         2         3
# 0 NaN  0.11757  0.871754  0.817941
# 1 NaN      NaN  0.428440  0.366126
# 2 NaN      NaN       NaN  0.962865
# 3 NaN      NaN       NaN       NaN


upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))

drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > 0.90)]
cor_matrix[drop_list]
df.drop(drop_list, axis=1)


def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list


high_correlated_cols(df, plot=True)

drop_list = high_correlated_cols(df)

high_correlated_cols(df.drop(drop_list, axis=1), plot=True)


# Yaklaşık 600 mb'lık 400 civarı değişkenin olduğu bir veri setinde deneyelim.
# https://www.kaggle.com/c/ieee-fraud-detection/data?select=train_transaction.csv


df = pd.read_csv("datasets/fraud_train_transaction.csv")

df.head()
len(df.columns)

drop_list = high_correlated_cols(df, plot=True)

len(drop_list)

len(df.drop(drop_list, axis=1).columns)