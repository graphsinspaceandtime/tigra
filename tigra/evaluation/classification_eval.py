from numpy.ma.extras import average


def eval(embedding, classifier):
    scores_accuracy = []
    scores_precision = []
    scores_recall = []

    for e in embedding:
        for i in range(10):
            node_ids = e.get_node_ids()
            node_vectors = [e[node_id] for node_id in node_ids]
            labels = [e.get_label(node_id) for node_id in node_ids]

            (
                score_accuracy,
                score_precision,
                score_recall,
                score_f1,
            ) = classifier.evaluate(node_vectors, labels)

            scores_accuracy.append(score_accuracy)
            scores_precision.append(score_precision)
            scores_recall.append(score_recall)

    score_accuracy = average(scores_accuracy)
    score_precision = average(scores_precision)
    score_recall = average(scores_recall)
    score_f1 = 2 / (1 / score_precision + 1 / score_recall)

    return (score_accuracy, score_precision, score_recall, score_f1)
