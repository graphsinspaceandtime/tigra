import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


if __name__ == "__main__":

    dist_matrix = pd.read_csv("Plane_143_1_1.0_0.csv", header=None).values
    labels = pd.read_csv("Plane.csv", header=None).values.ravel()

    print(f"Dist matrix: {dist_matrix.shape}")
    print(f"Labels: {labels.shape}, Classes: {np.unique(labels)}")

    n = len(labels)
    idx = np.arange(n)

    idx_train, idx_test = train_test_split(
        idx, test_size=0.2, random_state=42, stratify=labels
    )

    y_train = labels[idx_train]
    y_test = labels[idx_test]

    D_train_train = dist_matrix[np.ix_(idx_train, idx_train)]
    D_test_train = dist_matrix[np.ix_(idx_test, idx_train)]

    print(f"\nTrain: {len(idx_train)} instances, Test: {len(idx_test)} instances")

    knn = KNeighborsClassifier(n_neighbors=1, metric="precomputed")
    knn.fit(D_train_train, y_train)

    y_pred = knn.predict(D_test_train)
    acc = accuracy_score(y_test, y_pred)

    print("1-NN DTW Accuracy:", acc)
