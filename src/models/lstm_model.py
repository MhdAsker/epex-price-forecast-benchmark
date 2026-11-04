import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class _LSTMNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        _, (h_n, _) = self.lstm(x)   # h_n: (1, batch, hidden_dim)
        out = self.fc(h_n.squeeze(0))  # (batch, 1)
        return out.squeeze(1)          # (batch,)


class LSTMModel:
    """
    Single-layer LSTM for hour-ahead price forecasting.
    Input : sequences of SEQUENCE_LEN=24 hours of scaled features
    Output: next-hour price prediction
    """

    SEQUENCE_LEN = 24

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.net = _LSTMNet(input_dim, hidden_dim)
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def _make_sequences(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Slide a window of SEQUENCE_LEN over X/y.
        X_seq: (n_samples, SEQUENCE_LEN, n_features)
        y_seq: (n_samples,)  — target at position t (the hour after the window)
        """
        seq_len = self.SEQUENCE_LEN
        n = len(X) - seq_len
        X_seq = np.stack([X[i : i + seq_len] for i in range(n)], axis=0)
        y_seq = y[seq_len:]
        return X_seq, y_seq

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 20,
        batch_size: int = 64,
        lr: float = 0.001,
    ) -> None:
        # Scale features — fit on train only
        X_train_sc = self.scaler.fit_transform(X_train)
        X_val_sc = self.scaler.transform(X_val)

        X_tr_seq, y_tr_seq = self._make_sequences(X_train_sc, np.asarray(y_train, dtype=np.float32))
        X_va_seq, y_va_seq = self._make_sequences(X_val_sc, np.asarray(y_val, dtype=np.float32))

        tr_X = torch.tensor(X_tr_seq, dtype=torch.float32)
        tr_y = torch.tensor(y_tr_seq, dtype=torch.float32)
        va_X = torch.tensor(X_va_seq, dtype=torch.float32).to(self.device)
        va_y = torch.tensor(y_va_seq, dtype=torch.float32).to(self.device)

        dataset = torch.utils.data.TensorDataset(tr_X, tr_y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        # MAE (L1) loss directly optimises the reported evaluation metric,
        # producing more robust predictions on the heavy-tailed energy price
        # distribution than MSE (which over-penalises rare spikes).
        criterion = nn.L1Loss()

        best_val_loss = float("inf")
        patience = 5
        epochs_no_improve = 0
        best_state = None

        for epoch in range(1, epochs + 1):
            self.net.train()
            train_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                preds = self.net(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(xb)
            train_loss /= len(tr_y)

            self.net.eval()
            with torch.no_grad():
                val_preds = self.net(va_X)
                val_loss = criterion(val_preds, va_y).item()

            if epoch % 5 == 0:
                logger.info(
                    "LSTM epoch %d/%d — train loss: %.4f, val loss: %.4f",
                    epoch, epochs, train_loss, val_loss,
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_state = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info("LSTM early stopping at epoch %d (patience=%d)", epoch, patience)
                    break

        # Restore best weights
        if best_state is not None:
            self.net.load_state_dict(best_state)

        out_path = MODELS_DIR / "lstm.pt"
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "scaler": self.scaler,
            },
            out_path,
        )
        logger.info("Saved LSTM model to %s (best val loss: %.4f)", out_path, best_val_loss)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Scale X, build sequences, run forward pass.
        Returns array of length len(X) - SEQUENCE_LEN.
        """
        X_sc = self.scaler.transform(X)
        # Dummy y for _make_sequences (values unused — we only need X_seq shape)
        dummy_y = np.zeros(len(X_sc), dtype=np.float32)
        X_seq, _ = self._make_sequences(X_sc, dummy_y)

        tensor = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
        self.net.eval()
        with torch.no_grad():
            preds = self.net(tensor).cpu().numpy()
        return preds
