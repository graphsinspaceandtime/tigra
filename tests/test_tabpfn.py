from tabpfn import TabPFNClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = TabPFNClassifier()
# To use TabPFNv2:
# model = TabPFNClassifier.create_default_for_version(ModelVersion.V2)
model.fit(X_train, y_train)

# Predict class labels
preds = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))

# Predict class probabilities
probs = model.predict_proba(X_test)
print("Log-Loss:", log_loss(y_test, probs))
