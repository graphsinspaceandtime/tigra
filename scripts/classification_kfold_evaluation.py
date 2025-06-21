import sys

from numpy.ma.extras import average

from tests.mars.utils import suppress_stdout

from sklearn import tree, ensemble
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import make_scorer, precision_score, recall_score, accuracy_score

sys.path.append("../..")

from tigra.embeddings.embedding import FSEmbedding

def evalCV(embedding):
    print("Classifier\tAccuracy\tPrecision\tRecall\tF1")

    avg = "weighted"

    scorers = {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(precision_score, average=avg, zero_division=0),
        "recall": make_scorer(recall_score, average=avg, zero_division=0)
    }
    
    for e in embedding:
        node_ids = e.get_node_ids()
        node_vectors = [e[node_id] for node_id in node_ids]
        labels = [e.get_label(node_id) for node_id in node_ids]

        models = [tree.DecisionTreeClassifier(), SVC(), KNeighborsClassifier(), GaussianNB(), ensemble.RandomForestClassifier()]
        names = ["DecisionTree", "SVM", "kNN", "NaiveBayes", "RandomForest"]

        scores = {name: {"accuracy": [], "precision": [], "recall": []} for name in names}

        for model, name in zip(models, names):
            
            cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

            acc = cross_val_score(model, node_vectors, labels, scoring=scorers["accuracy"], cv=cv).mean()
            prec = cross_val_score(model, node_vectors, labels, scoring=scorers["precision"], cv=cv).mean()
            rec = cross_val_score(model, node_vectors, labels, scoring=scorers["recall"], cv=cv).mean()

            scores[name]["accuracy"].append(acc)
            scores[name]["precision"].append(prec)
            scores[name]["recall"].append(rec)
            

    for name in names:
        acc_avg = average(scores[name]["accuracy"])
        prec_avg = average(scores[name]["precision"])
        rec_avg = average(scores[name]["recall"])
        f1_avg = 2 / (1 / prec_avg + 1 / rec_avg)
        print(f"{name}\t{acc_avg:.4f}\t{prec_avg:.4f}\t{rec_avg:.4f}\t{f1_avg:.4f}")


def main_eval():
    datasets = [
        "ia-primary-school-proximity-attr",
        "ia-hospital-ward-proximity-attr",
    ]

    for dataset in datasets:
        with suppress_stdout():
            # Load the dataset(s) with labels
            embedding = FSEmbedding("embs/deepLID/" + dataset)
        
        print(dataset)
        evalCV(embedding)
            

if __name__ == "__main__":
    main_eval()
