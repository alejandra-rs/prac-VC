import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def classification_metrics(y_true, y_pred, class_names):
    labels = list(range(len(class_names)))

    def f1(average):
        return f1_score(y_true, y_pred, labels=labels, average=average, zero_division=0)

    return {"accuracy": accuracy_score(y_true, y_pred),
            "f1_macro": f1("macro"),
            "f1_weighted": f1("weighted"),
            "f1_per_class": dict(zip(class_names, f1(None))),
            "report": classification_report(y_true, y_pred, labels=labels, target_names=class_names, digits=3,
                                            zero_division=0)}


def plot_confusion_matrix(y_true, y_pred, class_names, title, ax):
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))), normalize="true")
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, cbar=False,
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set(xlabel="Predicción", ylabel="Real", title=title)
    ax.figure.tight_layout()
