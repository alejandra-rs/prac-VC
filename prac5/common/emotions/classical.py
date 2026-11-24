import numpy as np
from joblib import Parallel, delayed
from skimage.feature import hog, local_binary_pattern
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC


def pixels(images):
    return images.reshape(len(images), -1) / 255


def hog_features(images):
    return np.array([hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)) for image in images])


def lbp_histograms(image, cell=8):
    codes = local_binary_pattern(image, 8, 1, "uniform").astype(int)
    n = len(image) // cell
    cells = codes.reshape(n, cell, n, cell).swapaxes(1, 2).reshape(n * n, cell * cell)
    return np.concatenate([np.bincount(codes_in_cell, minlength=10) for codes_in_cell in cells]) / cell ** 2


def lbp_features(images):
    return np.array([lbp_histograms(image) for image in images])


def feature_extractor(name, pca_variance, seed):
    return {
        "pixels": make_pipeline(FunctionTransformer(pixels)),
        "eigenfaces": make_pipeline(FunctionTransformer(pixels), PCA(pca_variance, whiten=True, random_state=seed)),
        "fisherfaces": make_pipeline(FunctionTransformer(pixels), PCA(pca_variance, random_state=seed),
                                     LinearDiscriminantAnalysis()),
        "hog": make_pipeline(FunctionTransformer(hog_features), StandardScaler()),
        "lbp": make_pipeline(FunctionTransformer(lbp_features), StandardScaler()),
    }[name]


def svm(c, gamma):
    return SVC(C=c, gamma=gamma, class_weight="balanced")


def classical_model(name, c, gamma, pca_variance, seed):
    return make_pipeline(feature_extractor(name, pca_variance, seed), svm(c, gamma))


def tune_svm(extractor, train_images, train_labels, val_images, val_labels, grid_c, grid_gamma):
    train = extractor.fit_transform(train_images, train_labels)
    val = extractor.transform(val_images)
    default_gamma = 1 / (train.shape[1] * train.var())
    grid = [(c, g * default_gamma) for c in grid_c for g in grid_gamma]

    def score(c, gamma):
        return f1_score(val_labels, svm(c, gamma).fit(train, train_labels).predict(val), average="macro")

    scores = Parallel(n_jobs=len(grid), prefer="threads")(delayed(score)(c, gamma) for c, gamma in grid)
    return grid[int(np.argmax(scores))], np.reshape(scores, (len(grid_c), len(grid_gamma)))
