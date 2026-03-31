import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

def save_confusion_matrix(cm, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    cm_path = "reports/confusion_matrix.npy"
    output_path = "reports/figures/confusion_matrix.png"

    if not os.path.exists(cm_path):
        raise FileNotFoundError(f"{cm_path} not found.")

    cm = np.load(cm_path)
    save_confusion_matrix(cm, output_path)

    print(f"Confusion matrix saved to {output_path}")

if __name__ == "__main__":
    main()