import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

mnist = fetch_openml('mnist_784')  # Retrieves the dataset, nothing special


# Function to see the images stored in the Dataset above named "mnist"

def showimage(dframe, index):
    """Expects a DataFrame and the index of the photograph as an argument, converts to numpy and reshapes the image, then prints"""
    the_image = dframe.to_numpy()[index]
    the_image_resized = the_image.reshape(28,28)
    plt.imshow(the_image_resized, cmap='binary')
    plt.axis("off")
    plt.show()


train_img, test_img, train_lbl, test_lbl = train_test_split(mnist.data, mnist.target, test_size=1/7.0, random_state = 0)
test_img_copy = test_img.copy()  # Copied the test image DataFrame in order
# to compare them later as they will be changed in the future with transform operations




# Scaling the train and test image to prepare them for PCA operations
scaler = StandardScaler()
scaler.fit(train_img)  # Standardization process complete, now onto transforming

train_img = scaler.transform(train_img)  # This applies the standardization process done above.
test_img = scaler.transform(test_img)

# PCA operations. The goal is to protect the variance score atleast 95%
pca = PCA(.95)

pca.fit(train_img)

train_img = pca.transform(train_img)  # Applied
test_img = pca.transform(test_img)  # Applied


# Logistic Regression Operations
logisticRegr = LogisticRegression(solver = 'lbfgs', max_iter = 10000)  # 'lbfgs' is faster than the default solver.

logisticRegr.fit(train_img, train_lbl)
print(logisticRegr.predict(test_img[31].reshape(1,-1)))
print(showimage(test_img_copy, 31))

print(logisticRegr.score(test_img, test_lbl))