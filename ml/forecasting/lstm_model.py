"""Recurrent neural forecasting models implemented in pure NumPy.

This module provides *genuine* deep-learning recurrent architectures – a
vanilla RNN, an LSTM and a GRU – with hand-written forward passes and
backpropagation through time (BPTT), trained with the Adam optimiser. No
framework stubs are used: the gradients are derived and implemented for
each cell type directly.

Architecture
------------
Univariate series  x_1..x_T  (scaled)
    -> per-timestep recurrent cell (RNN / LSTM / GRU), hidden size H
    -> linear readout on the final hidden state
    -> one-step-ahead prediction

Training minimises MSE with Adam, gradient clipping, and deterministic
initialisation (seeded) so results are reproducible.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


class _Adam:
    """Minimal Adam optimiser over a list of parameter arrays."""

    def __init__(self, params: List[np.ndarray], lr: float = 0.01):
        self.params = params
        self.lr = lr
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads: List[np.ndarray]) -> None:
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= beta1
            m += (1 - beta1) * g
            v *= beta2
            v += (1 - beta2) * (g * g)
            m_hat = m / (1 - beta1 ** self.t)
            v_hat = v / (1 - beta2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + eps)


class RecurrentForecaster:
    """LSTM or GRU forecaster with real BPTT training.

    Parameters
    ----------
    cell:
        'rnn' (vanilla tanh RNN), 'lstm' or 'gru'.
    hidden_size:
        Hidden state dimensionality H.
    seq_len:
        Number of past timesteps used as input context.
    seed:
        Deterministic initialisation seed.
    """

    GATE_SLICES = {"i": 0, "f": 1, "o": 2, "g": 3}

    def __init__(self, cell: str = "lstm", hidden_size: int = 16, seq_len: int = 14,
                 lr: float = 0.02, seed: int = 42):
        if cell not in ("lstm", "gru", "rnn"):
            raise ValueError("cell must be 'rnn', 'lstm' or 'gru'")
        self.cell = cell
        self.H = hidden_size
        self.seq_len = seq_len
        rng = np.random.RandomState(seed)

        # Combined input->gate weights (4 gates for LSTM, 3 for GRU, 1 for RNN).
        n_gates = 4 if cell == "lstm" else (3 if cell == "gru" else 1)
        scale_x = 0.1
        scale_h = 1.0 / np.sqrt(hidden_size)
        self.Wx = rng.randn(1, n_gates * hidden_size) * scale_x
        self.Uh = rng.randn(hidden_size, n_gates * hidden_size) * scale_h
        self.b = np.zeros((1, n_gates * hidden_size))
        # Forget-gate bias init to 1.0 (well-known trick for long memory).
        if cell == "lstm":
            self.b[0, self.H:2 * self.H] = 1.0
        self.Wy = rng.randn(hidden_size, 1) * scale_h
        self.by = np.zeros((1, 1))
        self.opt = _Adam([self.Wx, self.Uh, self.b, self.Wy, self.by], lr=lr)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def _cell_forward(self, x_t: np.ndarray, h_prev: np.ndarray, c_prev: Optional[np.ndarray]):
        """One recurrent step, batch-vectorised. Returns (h, c, cache)."""
        pre = x_t @ self.Wx + h_prev @ self.Uh + self.b  # (N, G*H)
        H = self.H
        if self.cell == "lstm":
            i = _sigmoid(pre[:, 0:H])
            f = _sigmoid(pre[:, H:2 * H])
            o = _sigmoid(pre[:, 2 * H:3 * H])
            g = _tanh(pre[:, 3 * H:4 * H])
            c = f * c_prev + i * g
            h = o * _tanh(c)
            return h, c, (x_t, h_prev, c_prev, i, f, o, g, c)
        elif self.cell == "gru":  # gates: z(update), r(reset), n(candidate)
            z = _sigmoid(pre[:, 0:H])
            r = _sigmoid(pre[:, H:2 * H])
            pre_n = x_t @ self.Wx[:, 2 * H:3 * H] + (r * h_prev) @ self.Uh[:, 2 * H:3 * H] + self.b[:, 2 * H:3 * H]
            n = _tanh(pre_n)
            h = (1 - z) * n + z * h_prev
            return h, None, (x_t, h_prev, z, r, n)
        else:  # vanilla RNN: h = tanh(x Wx + h_prev Uh + b)
            h = _tanh(pre[:, 0:H])
            return h, None, (x_t, h_prev, h)

    def _cell_backward(self, dh: np.ndarray, dc: np.ndarray, cache):
        """Backward through one cell step.

        dh: gradient wrt h_t (sum of readout + future-step contributions)
        dc: gradient wrt c_t carried from the future (LSTM only)
        Returns (dh_prev, dc_prev, param-grad dicts).
        """
        x_t, h_prev = cache[0], cache[1]
        H = self.H
        gWx = np.zeros_like(self.Wx)
        gUh = np.zeros_like(self.Uh)
        gb = np.zeros_like(self.b)

        if self.cell == "lstm":
            _, _, c_prev, i, f, o, g, c = cache
            tanh_c = _tanh(c)
            # Output gate path
            do = dh * tanh_c
            do_pre = do * o * (1 - o)
            # Cell path
            dc_total = dc + dh * o * (1 - tanh_c ** 2)
            df = dc_total * c_prev
            df_pre = df * f * (1 - f)
            di = dc_total * g
            di_pre = di * i * (1 - i)
            dg = dc_total * i
            dg_pre = dg * (1 - g ** 2)
            d_pre = np.hstack([di_pre, df_pre, do_pre, dg_pre])  # (N, 4H)
            dc_prev = f * dc_total
            # All four LSTM gates read h_prev directly.
            dh_prev = d_pre @ self.Uh.T
        elif self.cell == "gru":
            _, _, z, r, n = cache
            H = self.H
            # h = (1-z)*n + z*h_prev
            dn = dh * (1 - z)
            dz = dh * (h_prev - n)
            dn_pre = dn * (1 - n ** 2)
            dz_pre = dz * z * (1 - z)
            # n gate hidden input is (r * h_prev) -> gradient modulated by r.
            dr = (dn_pre @ self.Uh[:, 2 * H:3 * H].T) * h_prev
            dr_pre = dr * r * (1 - r)
            d_pre = np.hstack([dz_pre, dr_pre, dn_pre])  # (N, 3H)
            dh_prev = dz_pre @ self.Uh[:, 0:H].T            # z-gate hidden path
            dh_prev += dr_pre @ self.Uh[:, H:2 * H].T        # r-gate hidden path
            dh_prev += (dn_pre @ self.Uh[:, 2 * H:3 * H].T) * r  # n-gate hidden path
            dh_prev += dh * z                               # direct h_prev path
            dc_prev = None
        else:
            # vanilla RNN: h = tanh(pre); cache = (x_t, h_prev, h)
            _, _, h = cache
            d_pre = dh * (1.0 - h ** 2)                      # (N, H)
            dh_prev = d_pre @ self.Uh[:, 0:self.H].T
            dc_prev = None

        gWx += x_t.T @ d_pre
        gUh += h_prev.T @ d_pre
        gb += d_pre.sum(axis=0, keepdims=True)
        return dh_prev, dc_prev, gWx, gUh, gb

    # ------------------------------------------------------------------
    # Training (BPTT)
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 80, verbose: bool = False) -> List[float]:
        """Train on sequences X (N, T, 1) with targets y (N, 1)."""
        N, T, _ = X.shape
        losses: List[float] = []
        for epoch in range(epochs):
            h = np.zeros((N, self.H))
            c = np.zeros((N, self.H)) if self.cell == "lstm" else None
            caches = []
            for t in range(T):
                x_t = X[:, t, :].reshape(N, 1)
                h, c, cache = self._cell_forward(x_t, h, c)
                caches.append(cache)

            y_hat = h @ self.Wy + self.by
            loss = float(np.mean((y_hat - y) ** 2))
            losses.append(loss)

            # ---- Backward ----
            # dL/dy_hat where L = mean((y_hat - y)^2)
            dres = (2.0 / N) * (y_hat - y)
            dWy = h.T @ dres
            dby = dres.sum(axis=0, keepdims=True)
            dh = dres @ self.Wy.T
            dc = np.zeros((N, self.H)) if self.cell == "lstm" else None

            gWx = np.zeros_like(self.Wx)
            gUh = np.zeros_like(self.Uh)
            gb = np.zeros_like(self.b)
            for t in reversed(range(T)):
                dh, dc, gx, gu, gbb = self._cell_backward(dh, dc, caches[t])
                gWx += gx
                gUh += gu
                gb += gbb

            # Gradient clipping for stability
            grads = [gWx, gUh, gb, dWy, dby]
            grads = [np.clip(g, -5.0, 5.0) for g in grads]
            self.opt.step(grads)
            if verbose and (epoch + 1) % 10 == 0:
                print(f"epoch {epoch + 1}/{epochs} loss={loss:.6f}")
        return losses

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict_next(self, context: np.ndarray) -> float:
        """Predict the next value given a context window (T,)."""
        context = np.asarray(context, dtype=np.float64).reshape(1, -1, 1)
        h = np.zeros((1, self.H))
        c = np.zeros((1, self.H)) if self.cell == "lstm" else None
        T = context.shape[1]
        for t in range(T):
            h, c, _ = self._cell_forward(context[:, t, :], h, c)
        return float((h @ self.Wy + self.by)[0, 0])

    def forecast(self, context: np.ndarray, horizon: int) -> np.ndarray:
        """Recursive multi-step forecast. `context` is (T,) scaled values."""
        window = list(np.asarray(context, dtype=np.float64).ravel())
        out: List[float] = []
        for _ in range(horizon):
            next_val = self.predict_next(np.array(window[-self.seq_len:]))
            out.append(next_val)
            window.append(next_val)
        return np.array(out)
