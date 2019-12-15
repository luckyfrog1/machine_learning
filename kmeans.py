from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
X, y = make_blobs(
    n_samples=150,
    n_features=2,
    centers=3,
    cluster_std=0.5,
    shuffle=True,
    random_state=0
)

km = KMeans(n_clusters=3,
            init='random',
            n_init=10,
            max_iter=100,
            tol=1e-04,
            random_state=0)


y_km = km.fit_predict(X)
