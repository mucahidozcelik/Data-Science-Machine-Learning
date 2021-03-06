############################################
# ASSOCIATION RULE LEARNING (BİRLİKTELİK KURALI ÖĞRENİMİ)
############################################

# Amacımız online retail II veri setine birliktelik analizi uygularak kullanıcılara ürün satın alma sürecinde
# ürün önermek

# 1. Veri Ön İşleme
# 2. ARL Veri Yapısını Hazırlama (Invoice-Product Matrix)
# 3. Birliktelik Kurallarının Çıkarılması
# 4. Çalışmanın Scriptini Hazırlama
# 5. Sepet Aşamasındaki Kullanıcılara Ürün Önerisinde Bulunmak

############################################
# Veri Ön İşleme
############################################


# !pip install mlxtend
import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

# Hemen verimizi hatırlayalım özlemişizdir.

df_ = pd.read_excel("datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()
df.info()
df.head()


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe

df = retail_data_prep(df)


############################################
# ARL Veri Yapısını Hazırlama (Invoice-Product Matrix)
############################################

# Verinin gelmesini istediğimiz durum:

# Description   NINE DRAWER OFFICE TIDY   SET 2 TEA TOWELS I LOVE LONDON    SPACEBOY BABY GIFT SET
# Invoice
# 536370                              0                                 1                       0
# 536852                              1                                 0                       1
# 536974                              0                                 0                       0
# 537065                              1                                 0                       0
# 537463                              0                                 0                       1


df_fr = df[df['Country'] == "France"]


df_fr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).head(20)

df_fr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().iloc[0:5, 0:5]

df_fr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().fillna(0).iloc[0:5, 0:5]

df_fr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().fillna(0).applymap(
    lambda x: 1 if x > 0 else 0).iloc[0:5, 0:5]

df_fr.groupby(['Invoice', 'Description']). \
    agg({"Quantity": "sum"}). \
    unstack(). \
    fillna(0). \
    applymap(lambda x: 1 if x > 0 else 0).iloc[0:5, 0:5]


def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)


fr_inv_pro_df = create_invoice_product_df(df_fr)
fr_inv_pro_df.head()

fr_inv_pro_df = create_invoice_product_df(df_fr, id=True)


def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)


check_id(df_fr, 21558)


############################################
# Birliktelik Kurallarının Çıkarılması
############################################

frequent_itemsets = apriori(fr_inv_pro_df, min_support=0.01, use_colnames=True)
# apriori invoice product matrisini verirsen bütün supportları(birlikte görünme olsılığı) hesaplar
# min support eşik değer belirler , bu olasılıktan düşük olan durumları dışarda bırakır

frequent_itemsets.sort_values("support", ascending=False).head(50)
# her bir ürünün bulunma olasılıklarını veriyor
rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
# association rules support değerlerini alıp confidence ve lift değerlerini bulur

rules.sort_values("support", ascending=False).head()
# antecedents : ilk ürün  consequents: sonraki ürün
# confidence:ilk ürünü alındığında 2.ürününde alınma olasılığı
# lift : ilk ürün alındığında 2.ürünün alınma olasılığı lift kadar artar
# leverage : support yüksek olan değerlere öncelik verme
# lift frekansı az da olsa örüntü verenleri yakalar o yüzden daha güvenilir yanlı değil ,
# leverage supportu yüksek olan değerlere öncelik verme eğiliminde o yüzden çok kullanılmıyor
#conviction : sonraki ürün olmadan öncü ürünün beklenen değeri

rules.sort_values("lift", ascending=False).head(500)



############################################
# Çalışmanın Scriptini Hazırlama
############################################

import pandas as pd

pd.set_option('display.max_columns', None)
from mlxtend.frequent_patterns import apriori, association_rules


# applymap tüm hücrelerde gezebilmek için kullanılır
# apply satır veya sütunlarda gezmeyi sağlar

def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)


def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)

def create_rules(dataframe, id=True, country="France"):
    dataframe = dataframe[dataframe['Country'] == country]
    dataframe = create_invoice_product_df(dataframe, id)
    frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
    return rules


df_ = pd.read_excel("datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()

df = retail_data_prep(df)
rules = create_rules(df)

# örnek olarak başka bir ülkeye göre yapalım.
rules_grm = create_rules(df, country="Germany")
rules_grm.sort_values("lift", ascending=False).head(50)


############################################
# Sepet Aşamasındaki Kullanıcılara Ürün Önerisinde Bulunmak
############################################


# Örnek:
# Kullanıcı örnek ürün id: 22492

product_id = 22492
check_id(df, product_id)

sorted_rules = rules.sort_values("lift", ascending=False)

recommendation_list = []
# tek tek ürünleri gezmek için liste oluşturduk
# sorted_rules da ürün setleri ve birbirleri ile olasılıkları var
# for döngüsü ; tüm ürünleri gez yazdığım product id yi bulursan dur rec_liste ,
# ilgili satırdaki consequents daki ilk ürünü seç
# listeye aldığımız ürün aslında lifti en yüksek olan ürün
# çünkü sorted rules da liftleri azalan sırayla sıraladığımız için [0] değerini girip en yükseğini seçtik


for i, product in enumerate(sorted_rules["antecedents"]):
    for j in list(product):
        if j == product_id:
            recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

recommendation_list[0:3]

check_id(df, 22556)

check_id(df, recommendation_list[0])


# rec_count kaç tane ürün önermek istediğin sayıyı yaz

def arl_recommender(rules_df, product_id, rec_count=1):

    sorted_rules = rules_df.sort_values("lift", ascending=False)

    recommendation_list = []

    for i, product in sorted_rules["antecedents"].items():
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))

    recommendation_list = list({item for item_list in recommendation_list for item in item_list})#ortaya çoklamayı yok etmek için

    return recommendation_list[:rec_count]

check_id(df, 23049)
arl_recommender(rules, 22492, 1)
arl_recommender(rules, 22492, 2)