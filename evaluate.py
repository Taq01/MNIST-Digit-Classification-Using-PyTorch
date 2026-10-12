import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from main import model, test_loader, criterion


def evaluate(model, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_true_labels = []
    all_predicted_labels = []

    with torch.no_grad():
        for images, true_labels in test_loader:
            logits = model(images)
            loss = criterion(logits, true_labels)
            predicted_labels = logits.argmax(dim=1)

            batch_size = true_labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            all_true_labels.extend(true_labels.cpu().tolist())
            all_predicted_labels.extend(predicted_labels.cpu().tolist())

    test_loss = total_loss / total_samples
    cm = confusion_matrix(all_true_labels, all_predicted_labels, labels=list(range(10)))
    test_accuracy = 100 * (cm.trace() / cm.sum())
    test_error = 100 - test_accuracy

    print("\nTest Results")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Test Error: {test_error:.2f}%")
    return cm


def plot_cm(cm):
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=list(range(10)),
    )
    fig, ax = plt.subplots(figsize=(9, 8))
    display.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    weights_path = "./weights/mnist_cnn_weights.pth"
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    cm = evaluate(model, test_loader, criterion)
    plot_cm(cm)
