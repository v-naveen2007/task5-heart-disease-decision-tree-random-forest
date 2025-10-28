import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    bc = load_breast_cancer()
    X = pd.DataFrame(bc.data, columns=bc.feature_names)
    y = pd.Series(bc.target, name="target")  # 0 = malignant, 1 = benign
    print("Dataset loaded from sklearn (breast cancer).")
    print("Shape:", X.shape)
    print("Target distribution:\n", y.value_counts(), "\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    dt = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)

    print("--- Decision Tree Results ---")
    print("Accuracy:", accuracy_score(y_test, y_pred_dt))
    print(classification_report(y_test, y_pred_dt))

    plt.figure(figsize=(18, 8))
    plot_tree(
        dt,
        filled=True,
        feature_names=X.columns,
        class_names=[str(c) for c in bc.target_names],
        rounded=True,
        fontsize=8,
    )
    plt.title("Decision Tree (max_depth=4)")
    plt.tight_layout()
    plt.savefig("decision_tree_plot.png", dpi=200)
    print("Saved decision tree visualization to decision_tree_plot.png")
    plt.close()

    rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    print("\n--- Random Forest Results ---")
    print("Accuracy:", accuracy_score(y_test, y_pred_rf))
    print(classification_report(y_test, y_pred_rf))

    fi = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    top_n = 12
    plt.figure(figsize=(10, 6))
    fi[:top_n].plot(kind="barh")
    plt.gca().invert_yaxis()
    plt.title(f"Top {top_n} Feature Importances (Random Forest)")
    plt.tight_layout()
    plt.savefig("feature_importances_rf.png", dpi=200)
    print("Saved feature importance plot to feature_importances_rf.png")
    plt.close()

    cv_dt = cross_val_score(dt, X, y, cv=5)
    cv_rf = cross_val_score(rf, X, y, cv=5)
    print("\nCross-Validation Mean Accuracy (Decision Tree): {:.4f}".format(cv_dt.mean()))
    print("Cross-Validation Mean Accuracy (Random Forest): {:.4f}".format(cv_rf.mean()))

    print("\nOverfitting Check:")
    print("Decision Tree -> Train: {:.3f}, Test: {:.3f}".format(dt.score(X_train, y_train), dt.score(X_test, y_test)))
    print("Random Forest -> Train: {:.3f}, Test: {:.3f}".format(rf.score(X_train, y_train), rf.score(X_test, y_test)))

    cm_dt = confusion_matrix(y_test, y_pred_dt)
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    print("\nConfusion matrix (Decision Tree):\n", cm_dt)
    print("\nConfusion matrix (Random Forest):\n", cm_rf)

if __name__ == "__main__":
    main()