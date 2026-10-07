from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True, as_frame=True)
print(X.shape)
print(y.value_counts())
print(X.head())