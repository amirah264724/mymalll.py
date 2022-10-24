import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

file = "mall_customer.csv"

import pandas as pd
df = pd.read_csv(file)

df.head()

features = ['Annual_Income_(k$)', 'Spending_Score']
X = df[features]

plt.scatter(X['Annual_Income_(k$)'], X['Spending_Score'])

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

X['Annual_Income_(k$)']

X

X['Spending_Score']

#plt.scatter(X[0], X[1], c=y_kmeans, s=50, cmap='viridis')
plt.scatter(X['Annual_Income_(k$)'], X['Spending_Score'], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
centers
#plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)

from sklearn.datasets import make_blobs
X, y_true = make_blobs(n_samples=200, centers=4,
                       cluster_std=0.60, random_state=0)
plt.scatter(X[:, 0], X[:, 1], s=50)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
