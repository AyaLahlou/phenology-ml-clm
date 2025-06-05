import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# Import the TemporalFusionTransformer from the local codebase
# This assumes a module defining the model exists in src.tft_model
# Replace the path below with the correct one if different
try:
    from src.tft_model import TemporalFusionTransformer
except ImportError:
    from src.model_training import TemporalFusionTransformer  # fallback if model defined here


# --------------------------- Dataset -----------------------------------------
class MMSDataset(Dataset):
    """Dataset for US MMS data prepared for TFT fine-tuning.

    We assume the CSV already contains windows of length `history_len` for each
    known attribute. Each attribute has columns formatted as
    `<attr>_0`, `<attr>_1`, ..., `<attr>_<history_len-1>`. Static attributes are
    provided once per row and the target column is `onset_doy`.
    """

    known_attrs = ['tmin', 'tmax', 'radiation', 'precipitation', 'sm', 'photoperiod']
    static_attrs = ['latitude', 'longitude']

    def __init__(self, csv_path: str, history_len: int = 90):
        super().__init__()
        self.history_len = history_len
        self.df = pd.read_csv(csv_path)
        # If there is a date column, ensure the dataframe is sorted
        if 'date' in self.df.columns:
            self.df = self.df.sort_values('date')

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        # collect historical known attributes
        ts_values = []
        for attr in self.known_attrs:
            for t in range(self.history_len):
                col = f"{attr}_{t}"
                ts_values.append(row[col])
        # reshape to (history_len, n_features)
        ts_array = np.array(ts_values, dtype=np.float32).reshape(self.history_len, -1)
        static_array = row[self.static_attrs].values.astype(np.float32)
        target = np.float32(row['onset_doy'])
        return (
            torch.from_numpy(ts_array),
            torch.from_numpy(static_array),
            torch.tensor(target)
        )


# --------------------------- Training Routine --------------------------------

def train(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (ts, static, target) in enumerate(dataloader):
        ts = ts.to(device)
        static = static.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(ts, static)
        loss = criterion(output.squeeze(), target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (batch_idx + 1) % 50 == 0:
            avg_loss = running_loss / 50
            print(f"Epoch {epoch}, Batch {batch_idx+1}, Loss = {avg_loss:.4f}")
            running_loss = 0.0


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for ts, static, target in dataloader:
            ts = ts.to(device)
            static = static.to(device)
            target = target.to(device)
            output = model(ts, static)
            loss = criterion(output.squeeze(), target)
            total_loss += loss.item()
            count += 1
    return total_loss / max(count, 1)


# --------------------------- Main -------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading dataset from", args.data_csv)
    dataset = MMSDataset(args.data_csv, history_len=90)

    # simple train/val split (80/20)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2)

    # Instantiate model
    model = TemporalFusionTransformer(
        seq_len=90,
        n_time_features=6,
        n_static_features=2,
        n_categorical_features=0,
        future_len=10,
    )
    checkpoint_path = args.checkpoint
    print(f"Loaded pretrained TFT from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Freeze all parameters except the output layer
    for param in model.parameters():
        param.requires_grad = False
    for param in model.output_layer.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    criterion = nn.MSELoss()

    print("Starting fine-tuning on US MMS dataset")
    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, optimizer, criterion, device, epoch)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch} validation loss: {val_loss:.4f}")

    # Save checkpoint
    model.eval()
    os.makedirs(os.path.dirname(args.output_checkpoint), exist_ok=True)
    torch.save(model.state_dict(), args.output_checkpoint)
    print(f"Saved fine-tuned model to {args.output_checkpoint}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune pretrained TFT on US MMS data")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to pretrained model checkpoint")
    parser.add_argument("--data_csv", type=str, required=True, help="Path to preprocessed MMS CSV file")
    parser.add_argument("--output_checkpoint", type=str, required=True, help="Output path for fine-tuned model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of fine-tuning epochs")
    args = parser.parse_args()

    main(args)
