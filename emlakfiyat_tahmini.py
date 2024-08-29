import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("evfiyatlari.csv", sep=";")#csv dosyasında noktalı virgül ile ayrıldığı için kullanıldı.
reg = linear_model.LinearRegression()
reg.fit(df[["alan","odasayisi","binayasi"]], df["fiyat"])
print(reg.predict([[230, 6, 0]]))#230 m2, 6 oda ve 0 yaşındaki binanın tahmini.

