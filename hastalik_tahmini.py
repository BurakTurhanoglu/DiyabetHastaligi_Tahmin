import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("diabetes.csv")
print(data.head())
# seker_hastalari = data[data.Outcome == 1]
# saglikli_insanlar = data[data.Outcome == 0]
# plt.scatter(saglikli_insanlar.Age, saglikli_insanlar.Glucose, color="green", label="sağlıklı", alpha = 0.4)
# plt.scatter(seker_hastalari.Age, seker_hastalari.Glucose, color="red", label="diyabet hastası", alpha = 0.4)
# plt.xlabel("Age")
# plt.ylabel("Glucose")
# plt.legend()
# plt.show()

y = data.Outcome.values
x_ham_veri = data.drop(["Outcome"], axis = 1)
#Normalizasyon işlemi.
x = (x_ham_veri - np.min(x_ham_veri)) / (np.max(x_ham_veri) - np.min(x_ham_veri))

#Eğitim ve Test verisinin ayrılması.
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state=1)

#KNN modeli oluşturulması.
knn = KNeighborsClassifier(n_neighbors = 6)
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)
print("K=3 değeri için Test verilerin doğrulama testi sonucu", knn.score(x_test, y_test))

#İdeal K değerinin belirlenmesi
def k_degeri():
    sayac = 1

    for k in range(1, 11):
        knn_yeni = KNeighborsClassifier(n_neighbors=k)
        knn_yeni.fit(x_train, y_train)
        print(f"{sayac}, Doğruluk oranı: % {knn_yeni.score(x_test, y_test) * 100}")
        sayac += 1
