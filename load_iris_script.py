from sklearn import datasets
import pandas as pd


iris = datasets.load_iris()

iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

iris_df.to_csv("datasets/iris.csv")
