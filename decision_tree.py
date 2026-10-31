# decision_tree.py - decision tree classification model and plots

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             log_loss, roc_auc_score, confusion_matrix)

VIEWPOINT = "Yes/no rules: hierarchically splits on presence of categories into transparent decision branches."


def train_decision_tree(x_train, y_train, x_test, y_test, labels=["Low", "Mid", "High"], seed=42):
    model = DecisionTreeClassifier(min_samples_leaf=5, random_state=seed)
    model.fit(x_train, y_train)

    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)
    prob_test = model.predict_proba(x_test)
    classes = list(model.classes_)

    metrics = {
        "Accuracy": accuracy_score(y_test, pred_test),
        "Precision": precision_score(y_test, pred_test, average="macro", zero_division=0),
        "Recall": recall_score(y_test, pred_test, average="macro", zero_division=0),
        "Macro F1": f1_score(y_test, pred_test, average="macro", zero_division=0),
        "Log Loss": log_loss(y_test, prob_test, labels=classes),
        "ROC-AUC": roc_auc_score(y_test, prob_test, multi_class="ovr", labels=classes),
        "cm": confusion_matrix(y_test, pred_test, labels=labels),
        "labels": labels,
        "Train Accuracy": accuracy_score(y_train, pred_train),
        "pred": pred_test,
        "prob": prob_test,
    }
    return model, metrics


def plot_confusion_matrix(cm, labels, title="Confusion Matrix (Decision Tree)"):
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                xticklabels=labels, yticklabels=labels, annot_kws={"size": 8})
    ax.set_title(title, fontsize=8, fontweight="bold")
    ax.set_xlabel("Predicted band", fontsize=8)
    ax.set_ylabel("Actual band", fontsize=8)
    fig.tight_layout()
    return fig


def plot_feature_importance(importance_series):
    fig, ax = plt.subplots(figsize=(3.8, 2.2))
    importance_series.sort_values().plot.barh(ax=ax, color="#0d9488", alpha=0.9)
    ax.set_title("Aggregated Feature Importance", fontsize=8, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig
