import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn import metrics
from sklearn.cluster import KMeans
import wandb


iris_data = pd.read_csv("datasets/iris.csv")

X = iris_data.drop(["target"],axis=1)
y = iris_data["target"]

wandb.init(

    project="iris clustering",
    
    config={
    # "learning_rate": 0.02,
    "experiment": "elbow",
    "architecture": "Kmeans",
    "dataset": "Foshers iris",
    "epochs": 1,
    "max_clusters": 100
    }
)

max_clusters = 100

for num_cluster in range(2,max_clusters):

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    kmeans = KMeans(n_clusters=num_cluster, n_init="auto").fit(X)

    labels = kmeans.predict(X)
    sumy = 0
    for i in range(len(X)):
        center = kmeans.cluster_centers_[labels[i]]
        sumy += sum((X.iloc[i] - center)**2/len(center))

    wandb.log({"Sum distance": sumy})