from alibi.confidence import TrustScore
from sklearn.decomposition import PCA


def reduce_dimensionality(X, pca = None, n_components = 0.95):
    if pca is None:
        pca = PCA(n_components=0.95)
        pca.fit(X.reshape(X.shape[0], X.shape[1] * X.shape[2]))
        X_dcd = pca.transform(X.reshape(X.shape[0], X.shape[1] * X.shape[2]))

        return pca, X_dcd
    else: 
        X_dcd = pca.transform(X.reshape(X.shape[0], X.shape[1] * X.shape[2]))

        return X_dcd


def train_ts_model(ts_model, X_train_dcd, Y_train, classes=2):
    ts_model.fit(X_train_dcd, Y_train, classes=classes)

    return ts_model