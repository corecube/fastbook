"""
A very simple model-training example — from scratch, no fine-tuning.

We train a tiny neural network (one hidden layer) to predict whether a
student is "admitted" (1) or not (0) from two inputs read out of a CSV file:
    hours_studied, hours_slept

Everything the "training" does is written out by hand with numpy:
  - initialise the weights
  - forward pass (make a prediction)
  - loss (how wrong we are)
  - backward pass (gradients, i.e. which way to nudge each weight)
  - gradient-descent step (nudge the weights)

There is no PyTorch, no fastai, no pretrained model. This is the whole
training loop, so you can read every line and change things to play with it.

Run it:   python3 train.py
Requires: numpy   (pip install numpy)
"""

import csv
import numpy as np

# ---------------------------------------------------------------------------
# Knobs to play with  ??  change these and re-run to see what happens
# ---------------------------------------------------------------------------
HIDDEN_UNITS  = 8       # size of the hidden layer (try 1, 4, 16, ...)
LEARNING_RATE = 0.1     # how big a step we take each update (try 0.01, 1.0)
EPOCHS        = 2000    # how many times we pass over the data (try 200, 5000)
SEED          = 0       # random seed so results are reproducible


# ---------------------------------------------------------------------------
# 1. Load the data from the CSV file
# ---------------------------------------------------------------------------
def load_csv(path):
    """Read the CSV into an input matrix X and a target vector y."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    X = np.array([[float(r["hours_studied"]), float(r["hours_slept"])] for r in rows])
    y = np.array([[float(r["admitted"])] for r in rows])   # shape (n, 1)
    return X, y


# ---------------------------------------------------------------------------
# 2. Little building blocks
# ---------------------------------------------------------------------------
def sigmoid(z):
    """Squash any number into the range (0, 1) so we can read it as a probability."""
    return 1.0 / (1.0 + np.exp(-z))


def normalize(X):
    """Rescale each column to mean 0, std 1. Training is much easier this way."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / std, mean, std


# ---------------------------------------------------------------------------
# 3. Train
# ---------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(SEED)

    X, y = load_csv("data.csv")
    X, mean, std = normalize(X)
    n_samples, n_features = X.shape

    # --- initialise the model's weights (this is "from scratch") ---
    # Layer 1:  inputs -> hidden
    W1 = rng.normal(0, 0.5, size=(n_features, HIDDEN_UNITS))
    b1 = np.zeros((1, HIDDEN_UNITS))
    # Layer 2:  hidden -> single output
    W2 = rng.normal(0, 0.5, size=(HIDDEN_UNITS, 1))
    b2 = np.zeros((1, 1))

    for epoch in range(EPOCHS):
        # ---- forward pass: make predictions ----
        z1 = X @ W1 + b1          # linear combination
        a1 = np.maximum(0, z1)    # ReLU activation (the non-linearity)
        z2 = a1 @ W2 + b2
        preds = sigmoid(z2)       # probability of "admitted"

        # ---- loss: binary cross-entropy (how wrong are we, on average) ----
        eps = 1e-8  # avoids log(0)
        loss = -np.mean(y * np.log(preds + eps) + (1 - y) * np.log(1 - preds + eps))

        # ---- backward pass: gradients via the chain rule ----
        dz2 = (preds - y) / n_samples      # gradient at the output
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0, keepdims=True)

        da1 = dz2 @ W2.T
        dz1 = da1 * (z1 > 0)               # ReLU passes gradient only where z1 > 0
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0, keepdims=True)

        # ---- gradient-descent step: nudge every weight downhill ----
        W2 -= LEARNING_RATE * dW2
        b2 -= LEARNING_RATE * db2
        W1 -= LEARNING_RATE * dW1
        b1 -= LEARNING_RATE * db1

        # ---- report progress ----
        if epoch % 200 == 0 or epoch == EPOCHS - 1:
            acc = ((preds > 0.5) == (y > 0.5)).mean()
            print(f"epoch {epoch:4d}   loss {loss:.4f}   accuracy {acc:.2%}")

    print("\nDone. Try a few predictions:")
    for hs, hl in [(8, 7), (2, 4), (5, 6), (1, 8)]:
        x = (np.array([[hs, hl]]) - mean) / std
        a1 = np.maximum(0, x @ W1 + b1)
        p = sigmoid(a1 @ W2 + b2)[0, 0]
        print(f"  studied {hs}h, slept {hl}h  ->  admitted prob {p:.2%}")


if __name__ == "__main__":
    main()
