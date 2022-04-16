###############################################
# ÇALIŞMA ORTAMI AYARLARI (SETTING UP WORKING ENVIRONMENT)
###############################################
# - PyCharm
# - Virtual Environments (conda)
# - Dependency Management (conda, pip)
# - Python İlk Adımlar


###############################################
# PYTHON İLK ADIMLAR
###############################################
# - Sayılar (Numbers) ve Karakter Dizileri (Strings)
# - Atamalar ve Değişkenler (Assignments & Variables)
# - Yazdırma Türleri (Print Types)
# - Kullanıcıdan Bilgi Almak: Input

###############################################
# Sayılar (Numbers) ve Karakter Dizileri (Strings)
###############################################

# string
print("Hello AI Era")

# integer
9

# float
9.2

# types
type(9)
type(9.2)
type("123")

###############################################
# Atamalar ve Değişkenler (Assignments & Variables)
###############################################

a = 9
b = 10
a * b
b - a
a * 5

hi = "Hello AI Era"

# del hi

###############################################
# Yazdırma Türleri (Print Types)
###############################################

# print
print("hello ai era")

name = "Rode"
age = 35

print(name, age)

# %
"Name: %s" % name
"Name: %s. Age: %s" % (name, age)

# str.format()
"Name: {}. Age: {}".format(name, age)

person = {"name": "Rode", "age": 35}

"Name:{}. Age: {}".format(person["name"], person["age"])

# fstring
print(f"Name: {name} Age: {age}")

###############################################
# Kullanıcıdan Bilgi Almak: Input
###############################################

number = input()
type(number)

number * 3
number / 3

int(number) / 3

num1 = int(input())
num2 = int(input())
num1 * num2

###############################################
# VERİ YAPILARI (DATA STRUCTURES)
###############################################

# - Sayılar (Numbers): int, float, complex
# - Karakter Dizileri (Strings): str
# - Boolean (TRUE-FALSE): bool
# - Liste (List)
# - Sözlük (Dictionary)
# - Demet (Tuple)
# - Set

###############################################
# Veri Yapılarına Giriş ve Hızlı Özet
###############################################

# Sayılar: integer
x = 46
type(x)

# Sayılar: float
x = 10.3
type(x)

# Sayılar: complex
x = 2j + 1
type(x)

# String
x = "Hello ai era"
type(x)

# Boolean
True
False
type(True)
5 == 4
3 == 2
1 == 1
type(3 == 2)
type(1 == 1)

# Liste
x = ["btc", "eth", "xrp", "ytd"]
type(x)

# Sözlük (dictionary)
x = {"name": "Peter",
     "Age": 36}

type(x)

# Tuple
x = ("python", "ml", "ds")
type(x)

# Set
x = {"python", "ml", "ds"}
type(x)

# Not: Liste, tuple, set ve dictionary veri yapıları aynı zamanda Python Collections (Arrays) olarak geçmektedir.


###############################################
# Sayılar (Numbers): int, float, complex
###############################################

a = 5
b = 10.3
a * 3
a / 6
a * b / 10
a ** 2

#######################
# Tipleri değiştirmek
#######################

int(b)
float(a)
int(a * b / 10)

###############################################
# Karakter Dizileri (Strings)
###############################################

print("John")
print('John')

#######################
# Çok Satırlı Karakter Dizileri
#######################

"""Veri Yapıları: Hızlı Özet,
Sayılar (Numbers): int, float, complex,
Karakter Dizileri (Strings): str,
List, Dictionary, Tuple, Set,
Boolean (TRUE-FALSE): bool"""

#######################
# Karakter Dizilerinin Elemanlarına Erişmek
#######################

name = "John"

name[0]
name[3]
# name[4]


#######################
# Karakter Dizilerinde Slice İşlemi
#######################

name[0:2]

#######################
# String İçerisinde Karakter Sorgulamak
#######################

topics = """Veri Yapıları: Hızlı Özet,
Sayılar (Numbers): int, float, complex,
Karakter Dizileri (Strings): str,
List, Dictionary, Tuple, Set,
Boolean (TRUE-FALSE): bool"""

"bool" in topics

print("bool" in topics)

###############################################
# String Metodları
###############################################

dir(str)

#######################
# len
#######################

type(len)

type(dir)

name = "john"

len(name)
len("MVK")

len("1")

#######################
# upper() & lower(): küçük-büyük dönüşümleri
#######################

"mvk".upper()
"JOHN".lower()

#######################
# replace: karakter değiştirir
#######################

hi = "Hello AI Era"
hi.replace("l", "p")
hi = hi.replace("l", "p")

#######################
# split: böler
#######################

"Hello AI Era".split()

#######################
# strip: kırpar
#######################

" ofofo ".strip()
"ofofo".strip("o")

#######################
# capitalize: ilk harfi büyütür
#######################

"foo".capitalize()

#######################
# isalnum: alfabetik mi ya da nümerik mi kontrolü
#######################

"foo".isalnum()

#######################
# isnumeric: nümerik mi kontrolü yapar
#######################

"foo".isnumeric()
"99".isnumeric()

###############################################
# Liste (List)
###############################################

# - Değiştirilebilir
# - Sıralıdır. Index işlemleri yapılabilir.
# - Kapsayıcıdır.

notes = [1, 2, 3, 4]
type(notes)

names = ["a", "b", "v"]

not_nam = [1, 2, 3, "a", "b", True, [1, 2, 3]]

notes[0]

notes[0:3]

notes[0] = 99

not_nam[6][1]

###############################################
# Liste Metodları (List Methods)
###############################################

dir(list)

#######################
# len: builtin python fonksiyonu, boyut bilgisi.
#######################

len(notes)

#######################
# append: eleman ekler
#######################

notes.append(90)

#######################
# pop: indexe göre siler
#######################

notes.pop(0)

#######################
# insert: indexe ekler
#######################

notes.insert(2, 99)

###############################################
# Sözlük (Dictonary)
###############################################

# Değiştirilebilir.
# Sırasız. (3.7 sonra sıralı.)
# Kapsayıcı.


dictionary = {"REG": "Regression",
              "LOG": "Logistic Regression",
              "CART": "Classification and Reg"}

dictionary = {"REG": ["RMSE", 10],
              "LOG": ["MSE", 20],
              "CART": ["SSE", 30]}

dictionary = {"REG": 10,
              "LOG": 20,
              "CART": 30}

#######################
# Key Sorgulama
#######################

"REG" in dictionary

#######################
# Key'e Göre Value'ya Erişmek
#######################

dictionary["REG"]

dictionary.get("REG")

#######################
# Value Değiştirmek
#######################

dictionary["REG"] = 99

#######################
# Tüm Key'lere Erişmek
#######################

dictionary.keys()

#######################
# Tüm Value'lara Erişmek
#######################

dictionary.values()

#######################
# Tüm Çiftleri Tuple Halinde Listeye Çevirme
#######################

dictionary.items()

#######################
# Key-Value Değerini Güncellemek
#######################

dictionary.update({"REG": 11})

#######################
# Yeni Key-Value Eklemek
#######################

dictionary.update({"RF": 10})

#######################
# Key ile Bir Item'ın Silinmesi
#######################

dictionary.pop("RF")

###############################################
# Demet (Tuple)
###############################################

# Listelerin değişime kapalı kardeşidir.

# - Değiştirilemez.
# - Sıralıdır.
# - Kapsayıcıdır.

t = ("john", "mark", 1, 2)

t[0:3]

names[0] = "999"

t[0] = "999"

t = list(t)
t[0] = "999"
t = tuple(t)

###############################################
# Set
###############################################

# - Değiştirilebilir.
# - Sırasız + Eşsizdir.
# - Kapsayıcıdır.


#######################
# difference(): İki kümenin farkı
#######################

set1 = set([1, 3, 5])
set2 = set([1, 2, 3])

# set1'de olup set2'de olmayanlar.
set1.difference(set2)
set1 - set2

# set2'de olup set1'de olmayanlar.
set2.difference(set1)
set2 - set1

#######################
# symmetric_difference(): İki kümede de birbirlerine göre olmayanlar
#######################

set1.symmetric_difference(set2)

#######################
# intersection(): İki kümenin kesişimi
#######################

set1 = set([1, 3, 5])
set2 = set([1, 2, 3])
set1.intersection(set2)
set2.intersection(set1)
set1 & set2

#######################
# union(): İki kümenin birleşimi
#######################

set1.union(set2)

#######################
# isdisjoint(): İki kümenin kesişimi boş mu?
#######################

set1 = set([7, 8, 9])
set2 = set([5, 6, 7, 8, 9, 10])

set1.isdisjoint(set2)

#######################
# issubset(): Bir küme diğer kümenin alt kümesi mi?
#######################

set1.issubset(set2)

#######################
# isdisjoint(): Bir küme diğer kümeyi kapsıyor mu?
#######################

set2.issuperset(set1)

###############################################
# FONKSİYONLAR, KOŞULLAR, DÖNGÜLER, COMPREHENSIONS
###############################################
# - Fonksiyonlar (Functions)
# - Koşullar (Conditions)
# - Döngüler (Loops)
# - Comprehesions
# - Lambda, Map, Reduce, Filter


###############################################
# FONKSİYONLAR (FUNCTIONS)
###############################################

#######################
# Fonksiyon nedir?
#######################

# Belirli görevleri yerine getirmek yazılan kod parçalarıdır.

#######################
# Fonksiyon Okuryazarlığı
#######################

print("a")
# ?print

help(print)

print("a", "b", sep="__")


#######################
# Fonksiyon Tanımlama
#######################


def calculate(x):
    print(x*2)

calculate(5)

def summer():
    print("argumanım yok benim")

summer()

def summer(arg1, arg2):
    print(arg1 + arg2)

summer(9, 10)

#######################
# Docstring
#######################

def summer(arg1, arg2):
    """
    Sum of two numbers

    Parameters
    ----------
    arg1: int, float
    arg2: int, float

    Returns
    -------
    NoneType

    """
    print(arg1 + arg2)


summer(1, 9)

help(summer)


#######################
# Fonksiyonların Statement/Body Bölümü
#######################


# def function_name(parameters/arguments):
#     statements (function body)


def multiplication(a, b):
    c = a * b
    print(c)

multiplication(8, 9)


def say_hi():
    print("Merhaba")
    print("Hi")
    print("Hello")
    print("hej hej")
    print("SA")

say_hi()

list_store = []


def add_element(a, b):
    c = a * b
    list_store.append(c)
    print(list_store)

add_element(10, 9)
add_element(18, 1)
add_element(180, 1)

#######################
# Ön Tanımlı Argümanlar/Parametreler (Default Parameters/Arguments)
#######################


def divide(a, b=1):
    print(a / b)

divide(9)



#######################
# Ne Zaman Fonksiyon Yazma İhtiyacımız Olur?
#######################


# varm, moisture, charge

(56 + 15) / 80
(17 + 45) / 70
(17 + 45) / 70
(56 + 15) / 80

# DRY

def calculate(varm, moisture, charge):
    print((varm + moisture) / charge)


calculate(90, 12, 12)
calculate(12, 56, 67)


#######################
# Return: Fonksiyon Çıktılarını Girdi Olarak Kullanmak
#######################

# calculate(12, 56, 67) * 10

type(calculate(12, 56, 67))

def calculate(varm, moisture, charge):
    return int((varm + moisture) / charge)

calculate(12, 988, 67) * 10


# def all_calculation(varm, moisture, charge, p):
#     a = calculate(varm, moisture, charge)
#     b = standardization(a, p)
#     print(b * 10)


def standardization(a, p):
    return a * 10 / 100 * p * p

standardization(10, 9)



def all_calculation(varm, moisture, charge, p):
    a = calculate(varm, moisture, charge)
    b = standardization(a, p)
    print(b * 10)

all_calculation(10, 90, 87, 10)


#######################
# Lokal & Global Değişkenler (Local & Global Variables)
#######################

list_store = [1, 2]

def add_element(a, b):
    c = a * b
    list_store.append(c)
    print(list_store)

add_element(10, 8)

#######################
# Doğru Fonksiyon Yazımı
#######################

# DRY (dont repeat yourself)
# DoT (Do one Thing)
# Modularity


###############################################
# KOŞULLAR (CONDITIONS)
###############################################

#######################
# if
#######################

1 == 1
1 == 2

if 1 == 1:
    print("something")


number = 10

if number == 10:
    print("10")

def number_check(number):
    if number == 10:
        print("equal to 10")


number_check(100)
number_check(10)

#######################
# else
#######################

def number_check(number):
    if number > 10:
        print("greater than 10")
    else:
        print("not greater than 10")


number_check(11)

#######################
# elif
#######################


def number_check(number):
    if number > 10:
        print("greater than 10")
    elif number < 10:
        print("less than 10")
    else:
        print("equal to 10")


number_check(10)

###############################################
# DÖNGÜLER (LOOPS)
###############################################

students = ["John", "Mark", "Venessa", "Mariam"]

students[0]
students[1]
students[2]
students[3]

for student in students:
    print(student)


for student in students:
    print(student + "_")

for student in students:
    print(f"Old Name: {student}, New Name: {student.upper()}")


salaries = [1000, 2000, 3000, 4000, 5000]

for i in salaries:
    print(i)


1000 * 20 / 100 + 1000
2000 * 20 / 100 + 2000
3000 * 20 / 100 + 3000

def new_salary(x):
    return x * 20 / 100 + x


new_salary(2000)


for salary in salaries:
    print(int(new_salary(salary)))


def raise_up(x):
    print(x * 10 / 100 + x)

def raise_down(x):
    print(x * 20 / 100 + x)


for salary in salaries:
    if salary >= 3000:
        raise_up(salary)
    else:
        raise_down(salary)



#######################
# break & continue & while
#######################

salaries = [1000, 2000, 3000, 4000, 5000]

for salary in salaries:
    if salary == 3000:
        print("break point")
        break
    print(salary)

for salary in salaries:
    if salary == 3000:
        continue
    print(salary)

number = 1

while number < 9:
    print(number)
    number += 1


#######################
# Mülakat Sorusu
#######################

# Amaç: Aşağıdaki şekilde string'i değiştirmek istiyoruz.

# before: "hi my name is john and i am learning python"
# after: "Hi mY NaMe iS JoHn aNd i aM LeArNiNg pYtHoN"

def alternating(string):
    new_string = ""

    for string_index in range(len(string)):
        if string_index % 2 == 0:
            new_string += string[string_index].upper()
        else:
            new_string += string[string_index].lower()

    print(new_string)

alternating("hi my name is john and i am learning python")


alternating("mvk")



#######################
# Enumerate: Otomatik Counter/Indexer ile for loop
#######################

# Problem: Listedeki öğrencileri index numaralarına göre iki gruba bölmek
students = ["John", "Mark", "Venessa", "Mariam"]


for student in students:
    print(student)

for index, student in enumerate(students):
    print(index, student)

A = []
B = []


for index, student in enumerate(students):
    if index % 2 == 0:
        A.append(student)
    else:
        B.append(student)
print(A)

#######################
# Mülakat Sorusu
#######################
# divide_students işlemini 2 grubu tek bir listede return edecek bir fonksiyonla yapınız.
students = ["John", "Mark", "Venessa", "Mariam"]

def divide_students(students):
    groups = [[], []]
    for index, student in enumerate(students):
        if index % 2 == 0:
            groups[0].append(student)
        else:
            groups[1].append(student)
    print(groups)
    return groups


divide_students(students)


#######################
# Enumerate'i Belirli Bir Index ile Kullanmak
#######################

for index, student in enumerate(students, 1):
    print(index, student)


def divide_students(students):
    groups = [[], []]
    for index, student in enumerate(students, 1):
        if index % 2 == 0:
            groups[0].append(student)
        else:
            groups[1].append(student)

    print(groups)


divide_students(students)


#######################
# Mülakat Sorusu
#######################

# Enumerate ile alternating sorusunun çözümü
# Alternating string with enumerate


def alternating_with_enumerate(string):
    new_string = ""

    for i, letter in enumerate(string):

        if i % 2 == 0:
            new_string += letter.upper()
        else:
            new_string += letter.lower()
    print(new_string)


alternating_with_enumerate("hi my name is john and i am learning python")

#######################
# Zip
#######################

students = ["John", "Mark", "Venessa", "Mariam"]

departments = ["mathematics", "statistics", "physics", "astronomy"]

ages = [23, 30, 26]

print(list(zip(students, departments, ages)))


###############################################
# lambda, map, filter, reduce
###############################################

def summer(a, b):
    return a + b

summer(1, 3) * 9

new_sum = lambda a, b: a + b
new_sum(1, 3) * 9

# MAP
salaries = [1000, 2000, 3000, 4000, 5000]


def new_salary(x):
    return x * 20 / 100 + x


new_salary(1000)

for salary in salaries:
    print(new_salary(salary))

list(map(new_salary, salaries))
list(map(lambda x: x * 20 / 100 + x, salaries))
list(map(lambda x: x ** 2, salaries))

list(map(lambda x: x.upper(), "john"))

# # FILTER
6 % 2 == 0

list_store = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
list(filter(lambda x: x % 2 == 0, list_store))

# # REDUCE
from functools import reduce
list_store = [1, 2, 3, 4]
reduce(lambda a, b: a + b, list_store)



###############################################
# COMPREHENSIONS
###############################################

#######################
# List Comprehension
#######################

salaries = [1000, 2000, 3000, 4000, 5000]

for salary in salaries:
    print(new_salary(salary))


null_list = []

for salary in salaries:
    null_list.append(new_salary(salary))



null_list = []

for salary in salaries:
    if salary > 3000:
        null_list.append(new_salary(salary))
    else:
        null_list.append(new_salary(salary * 2))

[new_salary(salary * 2) if salary < 3000 else new_salary(salary) for salary in salaries]


[salary * 2 for salary in salaries]

[salary * 2 for salary in salaries if salary < 3000]


[salary * 2 if salary < 3000 else salary * 0 for salary in salaries]

[new_salary(salary * 2) if salary < 3000 else new_salary(salary) for salary in salaries]


students = ["John", "Mark", "Venessa", "Mariam"]

students_no = ["John", "Venessa"]


[student.upper() if student not in students_no else student.lower() for student in students]

[student.lower() if student in students_no else student.upper() for student in students]

#######################
# Dictionary Comprehension
#######################

dictionary = {'a': 1,
              'b': 2,
              'c': 3,
              'd': 4}


dictionary["a"]


dictionary.keys()
dictionary.values()
dictionary.items()


for d in dictionary.values():
    print(d*2)

{k: v * 2 for (k, v) in dictionary.items()}

new_dict = {k: v ** 2 for (k, v) in dictionary.items()}

{k * 2: v for (k, v) in dictionary.items()}

{k.upper(): v for (k, v) in dictionary.items()}



#######################
# Döngü kullanarak sözlüğe eleman eklemek
#######################

#######################
# Mülakat Sorusu
#######################

# Amaç: çift sayıların karesi alınarak bir sözlüğe eklenmek istemektedir

numbers = range(10)
new_dict = {}

for n in numbers:
    if n % 2 == 0:
        new_dict[n] = n ** 2

new_dict


{n: n ** 2 for n in numbers if n % 2 == 0}



#######################
# List & Dict Comprehension Uygulamalar
#######################

#######################
# Bir Veri Setindeki Değişken İsimlerini Değiştirmek
#######################

# before:
# ['total', 'speeding', 'alcohol', 'not_distracted', 'no_previous', 'ins_premium', 'ins_losses', 'abbrev']

# after:
# ['TOTAL', 'SPEEDING', 'ALCOHOL', 'NOT_DISTRACTED', 'NO_PREVIOUS', 'INS_PREMIUM', 'INS_LOSSES', 'ABBREV']


import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

for col in df.columns:
    print(col.upper())

A = []

for col in df.columns:
    A.append(col.upper())

df.columns = A

df = sns.load_dataset("car_crashes")

df.columns = [col.upper() for col in df.columns]


#######################
# İsminde "INS" olan değişkenlerin başına FLAG diğerlerine NO_FLAG eklemek istiyoruz.
#######################

# before:
# ['TOTAL',
# 'SPEEDING',
# 'ALCOHOL',
# 'NOT_DISTRACTED',
# 'NO_PREVIOUS',
# 'INS_PREMIUM',
# 'INS_LOSSES',
# 'ABBREV']

# after:
# ['NO_FLAG_TOTAL',
#  'NO_FLAG_SPEEDING',
#  'NO_FLAG_ALCOHOL',
#  'NO_FLAG_NOT_DISTRACTED',
#  'NO_FLAG_NO_PREVIOUS',
#  'FLAG_INS_PREMIUM',
#  'FLAG_INS_LOSSES',
#  'NO_FLAG_ABBREV']



df.columns

[col for col in df.columns if "INS" in col]

["FLAG_" + col for col in df.columns if "INS" in col]


["FLAG_" + col if "INS" in col else "NO_FLAG_" + col for col in df.columns]

df.columns = ["FLAG_" + col if "INS" in col else "NO_FLAG_" + col for col in df.columns]



#######################
# Categorical Değişkenlerin Başına CAT yazmak.
#######################


# before:
# ['total',
# 'speeding',
# 'alcohol',
# 'not_distracted',
# 'no_previous',
# 'ins_premium',
# 'ins_losses',
# 'abbrev']

# after:
# ['TOTAL',
#  'SPEEDING',
#  'ALCOHOL',
#  'NOT_DISTRACTED',
#  'NO_PREVIOUS',
#  'INS_PREMIUM',
#  'INS_LOSSES',
#  'CAT_ABBREV']

df = sns.load_dataset("car_crashes")

[col for col in df.columns if df[col].dtype == "O"]


["CAT_" + col.upper() if df[col].dtype == "O" else col.upper() for col in df.columns]

# df.columns = ["CAT_" + col.upper() if df[col].dtype == "O" else col.upper() for col in df.columns]



# AŞAĞISI BİLGİMİZ OLSUN DİYE.

#######################
# Amaç key'i string, value'su aşağıdaki gibi bir liste olan sözlük oluşturmak.
#######################


# Output:
# {'total': ['mean', 'min', 'max', 'var'],
#  'speeding': ['mean', 'min', 'max', 'var'],
#  'alcohol': ['mean', 'min', 'max', 'var'],
#  'not_distracted': ['mean', 'min', 'max', 'var'],
#  'no_previous': ['mean', 'min', 'max', 'var'],
#  'ins_premium': ['mean', 'min', 'max', 'var'],
#  'ins_losses': ['mean', 'min', 'max', 'var']}


df = sns.load_dataset("car_crashes")
df.columns

num_cols = [col for col in df.columns if df[col].dtype != "O"]
soz = {}
agg_list = ["mean", "min", "max", "sum"]


for col in num_cols:
    soz[col] = agg_list


{col: agg_list for col in num_cols}


new_dict = {col: agg_list for col in num_cols}

df.groupby("abbrev").agg(new_dict)


df = sns.load_dataset("tips")

num_cols = [col for col in df.columns if df[col].dtype in [int, float]]


new_dict = {col: agg_list for col in num_cols}

df.groupby("time").agg(new_dict)


#######################
# Amaç: Value bölümündeki listenin her bir elemanını dinamik olarak biçimlendirmek
#######################
# Tüm sayısal değişkenlerin yanına mean , min ,max , var isimlerinden oluşan liste eklemek istiyoruz

# before:
# {'total': ['mean', 'min', 'max', 'sum'],
#  'speeding': ['mean', 'min', 'max', 'sum'],
#  'alcohol': ['mean', 'min', 'max', 'sum']}

# after:
# {'total': ['total_mean', 'total_min', 'total_max', 'total_var'],
#  'speeding': ['speeding_mean', 'speeding_min', 'speeding_max', 'speeding_var'],
#  'alcohol': ['alcohol_mean', 'alcohol_min', 'alcohol_max', 'alcohol_var']

import seaborn as sns
df = sns.load_dataset("car_crashes")

num_cols = [col for col in df.columns if df[col].dtype != "O"]

agg_list = ["mean", "min", "max", "sum"]

{col: [str(col) + "_" + c for c in agg_list] for col in num_cols}


###############################################
# AMAC: Bir listenin ilk elemanını key diğer elemanların tamamını da value olarak atamak
###############################################


# before
#    total  speeding  alcohol  not_distracted  no_previous
# 0   18.8     7.332    5.640          18.048       15.040
# 1   18.1     7.421    4.525          16.290       17.014
# 2   18.6     6.510    5.208          15.624       17.856
# 3   22.4     4.032    5.824          21.056       21.280
# 4   12.0     4.200    3.360          10.920       10.680


# after:
# {18.8: [7, 5, 18, 15],
#  18.1: [7, 4, 16, 17],
#  18.6: [6, 5, 15, 17],
#  22.4: [4, 5, 21, 21],
#  12.0: [4, 3, 10, 10]}

df = sns.load_dataset("car_crashes")
num_cols = [col for col in df.columns if df[col].dtype in [int, float]]

df[num_cols]

{row[0]: [int(s) for s in row[1:]] for row in df[num_cols].values}
