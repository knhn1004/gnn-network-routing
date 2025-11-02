"""Training script for MPNN shortest path prediction."""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

from gnn_routing.data import SyntheticGraphGenerator, create_training_pairs
from gnn_routing.models import MPNN


def get_device():
    """Get the best available device: CUDA > MPS > CPU.

    Returns:
        torch.device: The best available device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class ShortestPathDataset(Dataset):
    """Dataset for shortest path prediction."""

    def __init__(self, training_pairs):
        """Initialize dataset.

        Args:
            training_pairs: List of tuples (PyG Data object, shortest_path_distance)
        """
        self.data = training_pairs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data, target = self.data[idx]
        return data, target


def collate_fn(batch):
    """Custom collate function for PyG Data objects."""
    data_list, targets = zip(*batch)
    targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
    return data_list, targets


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch.

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Training", unit="batch", leave=False):
        data_list, targets = batch
        targets = targets.to(device)

        batch_loss = 0.0
        batch_size = len(data_list)

        for data, target in zip(data_list, targets):
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            batch_loss += loss.item()

        total_loss += batch_loss / batch_size
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(model, dataloader, criterion, device):
    """Validate model.

    Returns:
        Average loss and MAE for validation set
    """
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", unit="batch", leave=False):
            data_list, targets = batch
            targets = targets.to(device)

            for data, target in zip(data_list, targets):
                data = data.to(device)
                target = target.to(device)

                output = model(data)
                loss = criterion(output, target)
                mae = torch.abs(output - target).item()

                total_loss += loss.item()
                total_mae += mae
                num_samples += 1

    avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
    avg_mae = total_mae / num_samples if num_samples > 0 else 0.0

    return avg_loss, avg_mae


def main():
    parser = argparse.ArgumentParser(
        description="Train MPNN for shortest path prediction"
    )
    parser.add_argument(
        "--n_train_graphs", type=int, default=100, help="Number of training graphs"
    )
    parser.add_argument(
        "--n_val_graphs", type=int, default=20, help="Number of validation graphs"
    )
    parser.add_argument(
        "--n_pairs_per_graph",
        type=int,
        default=10,
        help="Number of (source, target) pairs per graph",
    )
    parser.add_argument(
        "--node_range",
        type=int,
        nargs=2,
        default=[100, 500],
        help="Range of node counts [min, max]",
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=64, help="Hidden dimension for GNN layers"
    )
    parser.add_argument(
        "--num_layers", type=int, default=3, help="Number of GNN layers"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device (CUDA > MPS > CPU)
    device = get_device()
    print(f"Using device: {device}")

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    # Generate training data
    print("Generating training graphs...")
    train_generator = SyntheticGraphGenerator(seed=args.seed)
    train_graphs = train_generator.generate_dataset(
        n_graphs=args.n_train_graphs, node_range=tuple(args.node_range)
    )

    print("Generating validation graphs...")
    val_generator = SyntheticGraphGenerator(seed=args.seed + 1)
    val_graphs = val_generator.generate_dataset(
        n_graphs=args.n_val_graphs, node_range=tuple(args.node_range)
    )

    # Create training pairs
    print("Creating training pairs...")
    train_pairs = create_training_pairs(
        train_graphs, n_pairs_per_graph=args.n_pairs_per_graph, seed=args.seed
    )

    print("Creating validation pairs...")
    val_pairs = create_training_pairs(
        val_graphs, n_pairs_per_graph=args.n_pairs_per_graph, seed=args.seed + 1
    )

    print(f"Training samples: {len(train_pairs)}")
    print(f"Validation samples: {len(val_pairs)}")

    # Create data loaders
    train_dataset = ShortestPathDataset(train_pairs)
    val_dataset = ShortestPathDataset(val_pairs)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Initialize model
    model = MPNN(
        node_feature_dim=4,
        edge_feature_dim=1,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        output_dim=1,
        dropout=0.1,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop with early stopping
    best_val_mae = float("inf")
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_maes = []

    print("\nStarting training...")
    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)

        # Validate
        val_loss, val_mae = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_maes.append(val_mae)

        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")
        print(f"  Val MAE: {val_mae:.6f}")

        # Checkpoint best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0

            # Save checkpoint
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_mae": val_mae,
                "args": vars(args),
            }
            torch.save(checkpoint, checkpoint_dir / "best_model.pt")
            print(f"  ✓ Saved best model (Val MAE: {val_mae:.6f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

        print()

    # Save training history
    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_maes": val_maes,
        "best_val_mae": best_val_mae,
    }

    with open(checkpoint_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining completed!")
    print(f"Best validation MAE: {best_val_mae:.6f}")
    print(f"Model saved to: {checkpoint_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
