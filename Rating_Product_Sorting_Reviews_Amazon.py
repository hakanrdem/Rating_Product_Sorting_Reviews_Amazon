# Rating Product &  Sorting Reviews in Amazon

# İş Problemi

"""
E-ticaretteki en önemli problemlerden bir tanesi ürünlere satış sonrası verilen puanların doğru
şekilde hesaplanmasıdır. Bu problemin çözümü e-ticaret sitesi için daha fazla müşteri memnuniyeti
sağlamak, satıcılar için ürünün öne çıkması ve satın alanlar için sorunsuz bir alışveriş deneyimi demektir.
Bir diğer problem ise ürünlere verilen yorumların doğru bir şekilde sıralanması olarak karşımıza çıkmaktadır.
Yanıltıcı yorumların öne çıkması ürünün satışını doğrudan etkileyeceğinden dolayı hem maddi kayıp hem de müşteri
kaybına neden olacaktır. Bu 2 temel problemin çözümünde e-ticaret sitesi ve satıcılar satışlarını arttırırken müşteriler
ise satın alma yolculuğunu sorunsuz olarak tamamlayacaktır.

"""

import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# Veri Seti Hikayesi

# Amazon ürün verilerini içeren bu veri seti ürün kategorileri ile çeşitli metadataları içermektedir.
# Elektronik kategorisindeki en fazla yorum alan ürünün kullanıcı puanları ve yorumları vardır.

# Proje Görevleri

# Görev 1
# Average Rating’i güncel yorumlara göre hesaplayınız ve var olan average rating ile kıyaslayınız.

# Paylaşılan veri setinde kullanıcılar bir ürüne puanlar vermiş ve yorumlar yapmıştır. Bu görevde amacımız verilen puanları tarihe göre
# ağırlıklandırarak değerlendirmek. İlk ortalama puan ile elde edilecek tarihe göre ağırlıklı puanın karşılaştırılması gerekmektedir.

# Adım1
# Ürünün ortalama puanını hesaplayınız.

df = pd.read_csv("dsmlbc_9_abdulkadir/Homeworks/hakan_erdem/3_Olcumleme_Problemleri/Rating_Product_Sorting_Reviews_Amazon/amazon_review.csv")
df.head(20)
df.shape

# Overall üzerinden rating dağılımı
df["overall"].value_counts()

# Overall üzerinden ortalama rating
df["overall"].mean()

df.groupby("helpful").agg({"helpful": "count",
                                   "overall": "mean"})

# Adım 2
# Tarihe göre ağırlıklı puan ortalamasını hesaplayınız.
# Adım 3
# Ağırlıklandırılmış puanlamada her bir zaman diliminin ortalamasını karşılaştırıp yorumlayınız.

# reviewTime değişkenini tarih değişkeni olarak tanıtmanız
df["reviewTime"] = pd.to_datetime(df["reviewTime"])

# reviewTime'ın max değerini current_date olarak kabul etmeniz
df["reviewTime"].max()  #  Timestamp('2014-12-07 00:00:00')
current_date=df["reviewTime"].max()

# her bir puan-yorum tarihi ile current_date'in farkını gün cinsinden ifade ederek yeni değişken oluşturmanız ve gün cinsinden ifade edilen
# değişkeni quantile fonksiyonu ile 4'e bölüp (3 çeyrek verilirse 4 parça çıkar) çeyrekliklerden gelen değerlere göre ağırlıklandırma yapmanız gerekir.
# Örneğin q1 = 12 ise ağırlıklandırırken 12 günden az süre önce yapılan yorumların ortalamasını alıp bunlara yüksek ağırlık vermek gibi.

df["days"] = (current_date - df["reviewTime"]).dt.days

p1,p2,p3,p4=df["days"].quantile([.1, .25, .5, .75])

df.loc[df["days"] <= p1, "overall"].mean()

df.loc[(df["days"] > p2) & (df["days"] <= p3), "overall"].mean()

df.loc[(df["days"] > p3) & (df["days"] <= p4), "overall"].mean()

df.loc[(df["days"] > p4), "overall"].mean()

def time_based_weighted_average(dataframe, w1=32, w2=22, w3=18, w4=17, w5=11):
    return dataframe.loc[df["days"] <= p1, "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["days"] > p1) & (dataframe["days"] <= p2), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["days"] > p2) & (dataframe["days"] <= p3), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["days"] > p3) & (dataframe["days"] <= p4), "overall"].mean() * w4 / 100+ \
           dataframe.loc[(dataframe["days"] > p4), "overall"].mean() * w5 / 100

time_based_weighted_average(df)

# Görev 2
# Ürün için ürün detay sayfasında görüntülenecek 20 review’i belirleyiniz.

# Adım 1
# helpful_no değişkenini üretiniz.

# total_vote bir yoruma verilen toplam up-down sayısıdır.
# up, helpful demektir.
# Veri setinde helpful_no değişkeni yoktur, var olan değişkenler üzerinden üretilmesi gerekmektedir.
# Toplam oy sayısından (total_vote) yararlı oy sayısı (helpful_yes) çıkarılarak yararlı bulunmayan oy sayılarını (helpful_no) bulunuz.

df.head(50)
df["helpful_yes"].value_counts()
df["helpful_no"]=df["total_vote"]-df["helpful_yes"]

# Adım 2
# score_pos_neg_diff, score_average_rating ve wilson_lower_bound skorlarını hesaplayıp veriye ekleyiniz.

def score_pos_neg_diff(up, down):
    return up - down

def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

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


df["score_pos_neg_diff"]=df.apply(lambda x:score_pos_neg_diff(x["helpful_yes"], x["helpful_no"]),axis=1)
df["score_average_rating"]=df.apply(lambda x:score_average_rating(x["helpful_yes"], x["helpful_no"]),axis=1)
df["wilson_lower_bound"]=df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"], confidence=0.95),axis=1)

df.head()

# Adım 3
# 20 Yorumu belirleyiniz ve sonuçları Yorumlayınız.
# wilson_lower_bound'a göre ilk 20 yorumu belirleyip sıralayanız.

df.sort_values("wilson_lower_bound", ascending=False).head(20)







