"""
4_cnn.py
Implementasi CNN dari Nol menggunakan NumPy
Komponen:
  - Convolution Layer (forward + backward)
  - Activation: ReLU, Sigmoid, Tanh
  - Max Pooling (forward + backward)
  - Flatten
  - Fully Connected Layer (forward + backward)
  - Binary Cross-Entropy Loss
  - SGD Optimizer
"""

import numpy as np


# ══════════════════════════════════════════════════════
# ACTIVATION FUNCTIONS
# ══════════════════════════════════════════════════════

class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x > 0)
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        return self.out

    def backward(self, dout):
        return dout * self.out * (1.0 - self.out)


class Tanh:
    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = np.tanh(x)
        return self.out

    def backward(self, dout):
        return dout * (1.0 - self.out ** 2)


def get_activation(nama: str):
    """Factory function untuk activation."""
    nama = nama.lower()
    if nama == "relu":
        return ReLU()
    elif nama == "sigmoid":
        return Sigmoid()
    elif nama == "tanh":
        return Tanh()
    else:
        raise ValueError(f"Activation '{nama}' tidak dikenal. Pilih: relu, sigmoid, tanh")



# ══════════════════════════════════════════════════════
# CONVOLUTION LAYER
# ══════════════════════════════════════════════════════

class ConvLayer:
    """
    Convolution 2D
    Input  : (N, H, W, C_in)
    Output : (N, H_out, W_out, C_out)
    """

    def __init__(self, n_filter: int, filter_size: int, n_channel: int,
                 stride: int = 1, padding: int = 0, lr: float = 0.01):
        self.n_filter   = n_filter
        self.filter_size = filter_size
        self.stride     = stride
        self.padding    = padding
        self.lr         = lr

        # Inisialisasi bobot dengan He initialization
        scale = np.sqrt(2.0 / (filter_size * filter_size * n_channel))
        self.W = np.random.randn(n_filter, filter_size, filter_size, n_channel) * scale
        self.b = np.zeros(n_filter)

        # Cache untuk backward
        self.x_pad = None
        self.x_col = None

    def _pad(self, x):
        if self.padding == 0:
            return x
        return np.pad(x, ((0,0),(self.padding, self.padding),
                           (self.padding, self.padding),(0,0)), mode="constant")

    def _im2col(self, x_pad, H_out, W_out):
        """
        Konversi input ke kolom matriks untuk efisiensi konvolusi.
        Output shape: (N * H_out * W_out, filter_size * filter_size * C)
        """
        N, H, W, C = x_pad.shape
        fs = self.filter_size
        s  = self.stride
        cols = []
        for i in range(H_out):
            for j in range(W_out):
                patch = x_pad[:, i*s:i*s+fs, j*s:j*s+fs, :]  # (N, fs, fs, C)
                cols.append(patch.reshape(N, -1))
        # cols: list of H_out*W_out arrays, each (N, fs*fs*C)
        col = np.stack(cols, axis=1)  # (N, H_out*W_out, fs*fs*C)
        return col.reshape(N * H_out * W_out, -1)

    def forward(self, x):
        N, H, W, C = x.shape
        fs = self.filter_size
        s  = self.stride
        p  = self.padding

        H_out = (H + 2*p - fs) // s + 1
        W_out = (W + 2*p - fs) // s + 1

        self.x_shape = x.shape
        self.H_out   = H_out
        self.W_out   = W_out
        self.N       = N

        x_pad      = self._pad(x)
        self.x_pad = x_pad

        # im2col: (N*H_out*W_out, fs*fs*C)
        x_col      = self._im2col(x_pad, H_out, W_out)
        self.x_col = x_col

        # W_col: (n_filter, fs*fs*C)
        W_col = self.W.reshape(self.n_filter, -1)

        # out_col: (N*H_out*W_out, n_filter)
        out_col = x_col @ W_col.T + self.b

        # Reshape ke (N, H_out, W_out, n_filter)
        out = out_col.reshape(N, H_out, W_out, self.n_filter)
        return out

    def backward(self, dout):
        N, H_out, W_out, nf = dout.shape
        fs = self.filter_size
        s  = self.stride
        p  = self.padding
        N_orig, H, W, C = self.x_shape

        # dout reshape ke (N*H_out*W_out, n_filter)
        dout_col = dout.reshape(-1, nf)

        # Gradient bias
        self.db = dout_col.sum(axis=0)

        # Gradient W
        # dW_col: (n_filter, fs*fs*C)
        dW_col   = dout_col.T @ self.x_col
        self.dW  = dW_col.reshape(self.W.shape)

        # Gradient input
        W_col  = self.W.reshape(nf, -1)
        dx_col = dout_col @ W_col  # (N*H_out*W_out, fs*fs*C)

        # col2im
        dx_pad = np.zeros_like(self.x_pad)
        dx_col_reshaped = dx_col.reshape(N, H_out * W_out, -1)
        idx = 0
        for i in range(H_out):
            for j in range(W_out):
                patch = dx_col_reshaped[:, idx, :].reshape(N, fs, fs, C)
                dx_pad[:, i*s:i*s+fs, j*s:j*s+fs, :] += patch
                idx += 1

        # Hapus padding
        if p > 0:
            dx = dx_pad[:, p:-p, p:-p, :]
        else:
            dx = dx_pad

        # Update bobot
        self.W -= self.lr * self.dW
        self.b -= self.lr * self.db

        return dx

# ══════════════════════════════════════════════════════
# ACTIVATION LAYER
# ══════════════════════════════════════════════════════

class ActivationLayer:
    

    def __init__(self, activation: str):
        self.activation_name = activation
        self.activation      = get_activation(activation)

    def forward(self, x):
        return self.activation.forward(x)

    def backward(self, dout):
        return self.activation.backward(dout)
    

# ══════════════════════════════════════════════════════
# MAX POOLING LAYER
# ══════════════════════════════════════════════════════

class MaxPooling:
    """
    Max Pooling 2D
    Input  : (N, H, W, C)
    Output : (N, H//pool_size, W//pool_size, C)
    """

    def __init__(self, pool_size: int = 2, stride: int = 2):
        self.pool_size = pool_size
        self.stride    = stride
        self.x         = None
        self.mask      = None

    def forward(self, x):
        N, H, W, C = x.shape
        ps = self.pool_size
        s  = self.stride

        H_out = (H - ps) // s + 1
        W_out = (W - ps) // s + 1

        self.x     = x
        self.H_out = H_out
        self.W_out = W_out

        out  = np.zeros((N, H_out, W_out, C))
        mask = np.zeros_like(x, dtype=bool)

        for i in range(H_out):
            for j in range(W_out):
                patch = x[:, i*s:i*s+ps, j*s:j*s+ps, :]  # (N, ps, ps, C)
                max_val = patch.max(axis=(1, 2), keepdims=True)
                out[:, i, j, :] = max_val[:, 0, 0, :]
                # Simpan posisi max untuk backward
                mask[:, i*s:i*s+ps, j*s:j*s+ps, :] |= (patch == max_val)

        self.mask = mask
        return out

    def backward(self, dout):
        N, H_out, W_out, C = dout.shape
        ps = self.pool_size
        s  = self.stride

        dx = np.zeros_like(self.x)

        for i in range(H_out):
            for j in range(W_out):
                d = dout[:, i, j, :][:, np.newaxis, np.newaxis, :]  # (N,1,1,C)
                patch_mask = self.mask[:, i*s:i*s+ps, j*s:j*s+ps, :]
                dx[:, i*s:i*s+ps, j*s:j*s+ps, :] += d * patch_mask

        return dx


# ══════════════════════════════════════════════════════
# FLATTEN LAYER
# ══════════════════════════════════════════════════════

class Flatten:
    def __init__(self):
        self.shape = None

    def forward(self, x):
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.shape)


# ══════════════════════════════════════════════════════
# FULLY CONNECTED LAYER
# ══════════════════════════════════════════════════════

class FullyConnected:
    """
    Dense Layer
    Input  : (N, input_dim)
    Output : (N, output_dim)
    """

    def __init__(self, input_dim: int, output_dim: int, lr: float = 0.01):
        scale    = np.sqrt(2.0 / input_dim)
        self.W   = np.random.randn(input_dim, output_dim) * scale
        self.b   = np.zeros(output_dim)
        self.lr  = lr
        self.x   = None

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        dx       = dout @ self.W.T
        self.dW  = self.x.T @ dout
        self.db  = dout.sum(axis=0)
        self.W  -= self.lr * self.dW
        self.b  -= self.lr * self.db
        return dx


# ══════════════════════════════════════════════════════
# OUTPUT LAYER + LOSS
# ══════════════════════════════════════════════════════

class SigmoidWithLoss:
    """
    Sigmoid + Binary Cross-Entropy Loss (digabung untuk stabilitas numerik)
    """

    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def forward(self, x, y_true):
        self.y_pred = 1.0 / (1.0 + np.exp(-np.clip(x.flatten(), -500, 500)))
        self.y_true = y_true.flatten().astype(np.float32)
        eps  = 1e-7
        loss = -np.mean(
            self.y_true * np.log(self.y_pred + eps) +
            (1 - self.y_true) * np.log(1 - self.y_pred + eps)
        )
        return loss

    def backward(self):
        N  = len(self.y_true)
        dx = (self.y_pred - self.y_true) / N
        return dx.reshape(-1, 1)

    def predict(self, x):
        prob = 1.0 / (1.0 + np.exp(-np.clip(x.flatten(), -500, 500)))
        return (prob >= 0.5).astype(int), prob


# ══════════════════════════════════════════════════════
# CNN MODEL
# ══════════════════════════════════════════════════════

class CNN:
    def __init__(self, n_conv: int = 1, activation: str = "relu",
                 lr: float = 0.01, use_pooling: bool = True,
                 input_shape: tuple = (64, 64, 3)):

        self.n_conv      = n_conv
        self.activation  = activation
        self.lr          = lr
        self.use_pooling = use_pooling
        self.layers      = []

        H, W, C = input_shape
        n_filter = 8  # filter pertama

        # ── Bangun Conv Blocks ──────────────────────────
        for i in range(n_conv):
            in_ch = C if i == 0 else n_filter * i

            # Pastikan in_ch tidak 0
            if i > 0:
                in_ch = n_filter * (2 ** (i - 1))

            out_ch = n_filter * (2 ** i)  # 8, 16, 32

            self.layers.append(ConvLayer(
                n_filter    = out_ch,
                filter_size = 3,
                n_channel   = C if i == 0 else n_filter * (2 ** (i - 1)),
                stride      = 1,
                padding     = 1,  # same padding → ukuran tetap
                lr          = lr
            ))
            self.layers.append(ActivationLayer(activation))

            if use_pooling:
                self.layers.append(MaxPooling(pool_size=2, stride=2))
                H = H // 2
                W = W // 2

            C = out_ch  # channel output jadi input layer berikutnya

        # ── Flatten ────────────────────────────────────
        self.layers.append(Flatten())
        fc_input_dim = H * W * C

        # ── Fully Connected ────────────────────────────
        self.layers.append(FullyConnected(fc_input_dim, 64, lr=lr))
        self.layers.append(ActivationLayer(activation))
        self.layers.append(FullyConnected(64, 1, lr=lr))

        # ── Loss ───────────────────────────────────────
        self.loss_layer = SigmoidWithLoss()

    def forward(self, x, y=None):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        if y is not None:
            loss = self.loss_layer.forward(out, y)
            return out, loss
        return out

    def backward(self):
        dout = self.loss_layer.backward()
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

    def predict(self, x):
        out = self.forward(x)
        pred, prob = self.loss_layer.predict(out)
        return pred, prob

    def accuracy(self, x, y):
        pred, _ = self.predict(x)
        return np.mean(pred == y.flatten())

    def get_weights(self) -> list:
        
        weights = []
        for layer in self.layers:
            if isinstance(layer, (ConvLayer, FullyConnected)):
                weights.append({
                    "W": layer.W.copy(),
                    "b": layer.b.copy(),
                })
            else:
                weights.append(None)  
        return weights

    def set_weights(self, weights: list):
        """
        Kembalikan bobot model ke snapshot tertentu.
        Parameter: list hasil get_weights()
        """
        idx = 0
        for layer in self.layers:
            if isinstance(layer, (ConvLayer, FullyConnected)):
                layer.W = weights[idx]["W"].copy()
                layer.b = weights[idx]["b"].copy()
            idx += 1


# ══════════════════════════════════════════════════════
# TRAINING FUNCTION
# ══════════════════════════════════════════════════════

def train(model: CNN, X_train: np.ndarray, y_train: np.ndarray,
          X_test: np.ndarray, y_test: np.ndarray,
          epochs: int = 20, batch_size: int = 32,
          patience: int = 5, lr_decay: bool = True,
          decay_every: int = 10, decay_factor: float = 0.5,
          verbose: bool = True):
    """
    Training loop dengan:
      - Mini-batch SGD
      - Gradient clipping
      - Best model checkpoint (simpan bobot epoch terbaik berdasarkan TEST)
      - Early stopping (berhenti jika test acc tidak naik)
      - Learning rate decay (turunkan LR setiap N epoch)

    Return
    ------
    history : dict berisi loss, train_acc, test_acc per epoch
              + best_epoch dan best_test_acc
    """
    history = {
        "loss"          : [],
        "train_acc"     : [],
        "test_acc"      : [],
        "best_epoch"    : 1,
        "best_test_acc" : 0.0,
    }

    N            = len(X_train)
    best_acc     = 0.0
    best_weights = model.get_weights()
    no_improve   = 0

    for epoch in range(1, epochs + 1):

        # ── Learning Rate Decay ──────────────────────
        if lr_decay and epoch > 1 and (epoch - 1) % decay_every == 0:
            for layer in model.layers:
                if isinstance(layer, (ConvLayer, FullyConnected)):
                    layer.lr *= decay_factor
            if verbose:
                cur_lr = model.layers[0].lr if hasattr(model.layers[0], "lr") else "?"
                print(f"   LR decay → LR baru: {cur_lr:.6f}")

        # ── Shuffle data ─────────────────────────────
        idx        = np.random.permutation(N)
        X_shuffled = X_train[idx]
        y_shuffled = y_train[idx]

        epoch_loss = 0.0
        n_batch    = 0

        # ── Mini-batch loop ──────────────────────────
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            X_b = X_shuffled[start:end]
            y_b = y_shuffled[start:end]

            _, loss = model.forward(X_b, y_b)
            epoch_loss += loss
            n_batch    += 1

            model.backward()

            # ── Gradient Clipping ────────────────────
            for layer in model.layers:
                if isinstance(layer, (ConvLayer, FullyConnected)):
                    np.clip(layer.dW, -1.0, 1.0, out=layer.dW)
                    np.clip(layer.db, -1.0, 1.0, out=layer.db)

        avg_loss  = epoch_loss / n_batch

        # Evaluasi di train (sample) dan test
        idx_sample = np.random.choice(N, min(300, N), replace=False)
        train_acc  = model.accuracy(X_train[idx_sample], y_train[idx_sample])
        test_acc   = model.accuracy(X_test, y_test)

        history["loss"].append(avg_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        # ── Best Model Checkpoint (berdasarkan test_acc) ──
        marker = ""
        if test_acc > best_acc:
            best_acc     = test_acc
            best_weights = model.get_weights()
            history["best_epoch"]    = epoch
            history["best_test_acc"] = best_acc
            no_improve = 0
            marker     = "    best"
        else:
            no_improve += 1
            marker      = f"  (no improve {no_improve}/{patience})"

        if verbose:
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Train Acc: {train_acc*100:.2f}% | "
                  f"Test Acc: {test_acc*100:.2f}%"
                  f"{marker}")

        # ── Early Stopping ───────────────────────────
        if no_improve >= patience:
            if verbose:
                print(f"\n   Early stopping di epoch {epoch} "
                      f"(tidak ada peningkatan selama {patience} epoch)")
            break

    # ── Restore bobot terbaik ─────────────────────────
    model.set_weights(best_weights)
    if verbose:
        print(f"\n   Bobot dikembalikan ke epoch terbaik: "
              f"epoch {history['best_epoch']} "
              f"(Test Acc: {best_acc*100:.2f}%)")

    return history

# ══════════════════════════════════════════════════════
# EVALUASI
# ══════════════════════════════════════════════════════

def confusion_matrix(model: CNN, X: np.ndarray, y: np.ndarray):
    """
    Hitung confusion matrix.
    Return: dict dengan TP, TN, FP, FN dan metrik turunannya
    """
    pred, _ = model.predict(X)
    y_flat  = y.flatten()

    TP = int(np.sum((pred == 1) & (y_flat == 1)))
    TN = int(np.sum((pred == 0) & (y_flat == 0)))
    FP = int(np.sum((pred == 1) & (y_flat == 0)))
    FN = int(np.sum((pred == 0) & (y_flat == 1)))

    total     = TP + TN + FP + FN
    accuracy  = (TP + TN) / total if total > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0)

    return {
        "matrix"    : np.array([[TN, FP], [FN, TP]]),
        "TP"        : TP, "TN": TN, "FP": FP, "FN": FN,
        "accuracy"  : accuracy,
        "precision" : precision,
        "recall"    : recall,
        "f1"        : f1,
    }