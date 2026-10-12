#AI generated
import base64
import io
import os
import random

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from PIL import Image
from flask import Flask, render_template

# Import the model class from main.py (project root is cwd)
from main import BasicCNN

# ---------- Paths (relative to project root, where you run -m webapp.app) ----------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "weights", "mnist_cnn_weights.pth")
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")

app = Flask(__name__, template_folder="templates")

# ---------- Load model once ----------
model = BasicCNN()
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
model.eval()

# ---------- Load test set once ----------
test_data = datasets.MNIST(
    root=DATA_DIR, train=False, download=True, transform=transforms.ToTensor()
)

def tensor_to_base64_png(tensor):
    arr = (tensor.squeeze(0).numpy() * 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

@app.route("/")
def index():
    idx = random.sample(range(len(test_data)), 10)

    images = torch.stack([test_data[i][0] for i in idx])   # (10,1,28,28)
    labels = [test_data[i][1] for i in idx]

    with torch.no_grad():
        logits = model(images)                             # (10,10)
        probs  = F.softmax(logits, dim=1)                  # (10,10)
        preds  = probs.argmax(dim=1).tolist()

    items = []
    for i in range(10):
        prob_list = probs[i].tolist()
        predicted = preds[i]
        items.append({
            "img": tensor_to_base64_png(images[i]),
            "actual": labels[i],
            "predicted": predicted,
            "correct": labels[i] == predicted,
            "probabilities": [
                {"digit": d, "percent": round(prob_list[d] * 100, 2)}
                for d in range(10)
            ],
            "confidence": round(prob_list[predicted] * 100, 2),
        })

    return render_template("index.html", items=items)

if __name__ == "__main__":
    app.run(debug=True)