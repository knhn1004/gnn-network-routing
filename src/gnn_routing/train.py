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
import wandb
from datetime import datetime
from dotenv import load_dotenv
import os

from gnn_routing.data import SyntheticGraphGenerator, create_training_pairs
from gnn_routing.models import create_model


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


def train_epoch(model, dataloader, optimizer, criterion, device, gradient_accumulation_steps=1, use_amp=False, scaler=None):
    """Train for one epoch with gradient accumulation and mixed precision support.

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training", unit="batch", leave=False)):
        data_list, targets = batch
        targets = targets.to(device)

        batch_loss = 0.0
        batch_size = len(data_list)

        for data, target in zip(data_list, targets):
            data = data.to(device)
            target = target.to(device)

            if use_amp and scaler is not None:
                with torch.cuda.amp.autocast():
                    output = model(data)
                    loss = criterion(output, target)
                    loss = loss / gradient_accumulation_steps
                
                scaler.scale(loss).backward()
                batch_loss += loss.item() * gradient_accumulation_steps
            else:
                output = model(data)
                loss = criterion(output, target)
                # Scale loss by gradient accumulation steps
                loss = loss / gradient_accumulation_steps
                loss.backward()
                batch_loss += loss.item() * gradient_accumulation_steps

        # Update weights every gradient_accumulation_steps batches
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if use_amp and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        total_loss += batch_loss / batch_size
        num_batches += 1

    # Handle remaining gradients if batch count is not divisible by gradient_accumulation_steps
    if num_batches % gradient_accumulation_steps != 0:
        if use_amp and scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

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
        "--dataset_multiplier",
        type=int,
        default=10,
        help="Multiplier for dataset size (default: 10x)",
    )
    parser.add_argument(
        "--use_augmentation",
        action="store_true",
        help="Use graph augmentation techniques (edge rewiring, weight perturbation, subgraph sampling)",
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
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps (effective batch size = batch_size * gradient_accumulation_steps)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of data loading workers (0 = main process only)",
    )
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
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="W&B project name (overrides env var WANDB_PROJECT)",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="W&B run name (overrides auto-generated name)",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="W&B logging mode",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="mpnn",
        choices=["mpnn", "gat"],
        help="Model architecture type",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=4,
        help="Number of attention heads (for GAT only)",
    )
    parser.add_argument(
        "--use_layer_norm",
        action="store_true",
        help="Use layer normalization (for GAT only)",
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="Use automatic mixed precision (FP16) training",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default=None,
        choices=[None, "cosine", "plateau"],
        help="Learning rate scheduler type",
    )
    parser.add_argument(
        "--include_graph_features",
        action="store_true",
        help="Include graph-level features (density, clustering, etc.)",
    )
    parser.add_argument(
        "--include_extra_centrality",
        action="store_true",
        help="Include additional centrality measures (closeness, eigenvector, PageRank)",
    )
    parser.add_argument(
        "--positional_encoding",
        type=str,
        default=None,
        choices=[None, "laplacian", "random_walk"],
        help="Type of positional encoding to use",
    )

    args = parser.parse_args()

    # Load environment variables from .env file
    load_dotenv()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Get wandb project name from env or args
    wandb_project = args.wandb_project or os.getenv(
        "WANDB_PROJECT", "gnn-network-routing"
    )

    # Generate run name: project_name + serialized datetime
    if args.wandb_run_name:
        wandb_run_name = args.wandb_run_name
    else:
        dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        wandb_run_name = f"{wandb_project}_{dt_str}"

    # Initialize wandb
    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        mode=args.wandb_mode,
        config=vars(args),
    )

    # Device (CUDA > MPS > CPU)
    device = get_device()
    print(f"Using device: {device}")
    wandb.config.update(
        {
            "system/device": str(device),
        }
    )

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    # Generate training data
    n_train_graphs_actual = args.n_train_graphs * args.dataset_multiplier
    print(f"Generating training graphs ({n_train_graphs_actual} graphs with {args.dataset_multiplier}x multiplier)...")
    train_generator = SyntheticGraphGenerator(seed=args.seed)
    train_graphs = train_generator.generate_dataset(
        n_graphs=n_train_graphs_actual,
        node_range=tuple(args.node_range),
        use_augmentation=args.use_augmentation,
    )

    print("Generating validation graphs...")
    val_generator = SyntheticGraphGenerator(seed=args.seed + 1)
    val_graphs = val_generator.generate_dataset(
        n_graphs=args.n_val_graphs, node_range=tuple(args.node_range)
    )
    
    # Log dataset statistics
    train_node_counts = [len(G) for G in train_graphs]
    train_edge_counts = [G.number_of_edges() for G in train_graphs]
    val_node_counts = [len(G) for G in val_graphs]
    val_edge_counts = [G.number_of_edges() for G in val_graphs]
    
    print(f"\nDataset Statistics:")
    print(f"  Training graphs: {len(train_graphs)}")
    print(f"    Node range: {min(train_node_counts)}-{max(train_node_counts)} (avg: {np.mean(train_node_counts):.1f})")
    print(f"    Edge range: {min(train_edge_counts)}-{max(train_edge_counts)} (avg: {np.mean(train_edge_counts):.1f})")
    print(f"  Validation graphs: {len(val_graphs)}")
    print(f"    Node range: {min(val_node_counts)}-{max(val_node_counts)} (avg: {np.mean(val_node_counts):.1f})")
    print(f"    Edge range: {min(val_edge_counts)}-{max(val_edge_counts)} (avg: {np.mean(val_edge_counts):.1f})")
    
    wandb.config.update(
        {
            "dataset/train_graphs": len(train_graphs),
            "dataset/val_graphs": len(val_graphs),
            "dataset/multiplier": args.dataset_multiplier,
            "dataset/use_augmentation": args.use_augmentation,
            "dataset/train_avg_nodes": float(np.mean(train_node_counts)),
            "dataset/train_avg_edges": float(np.mean(train_edge_counts)),
        }
    )

    # Create training pairs
    print("Creating training pairs...")
    train_pairs = create_training_pairs(
        train_graphs,
        n_pairs_per_graph=args.n_pairs_per_graph,
        seed=args.seed,
        include_graph_features=args.include_graph_features,
        include_extra_centrality=args.include_extra_centrality,
        positional_encoding=args.positional_encoding,
    )

    print("Creating validation pairs...")
    val_pairs = create_training_pairs(
        val_graphs,
        n_pairs_per_graph=args.n_pairs_per_graph,
        seed=args.seed + 1,
        include_graph_features=args.include_graph_features,
        include_extra_centrality=args.include_extra_centrality,
        positional_encoding=args.positional_encoding,
    )

    print(f"Training samples: {len(train_pairs)}")
    print(f"Validation samples: {len(val_pairs)}")

    # Auto-detect node feature dimension from first sample
    if len(train_pairs) > 0:
        sample_data, _ = train_pairs[0]
        node_feature_dim = sample_data.x.shape[1]
        print(f"Detected node feature dimension: {node_feature_dim}")
    else:
        # Fallback: calculate based on defaults
        # Base: 4 (degree, betweenness, is_source, is_target)
        # Extra centrality: +3 (closeness, eigenvector, PageRank) - default True
        # Graph features: +5 (density, clustering, path length, diameter, components) - default True
        # Positional encoding: +0 (default None)
        node_feature_dim = 4 + 3 + 5  # 12 features by default
        print(f"Using default node feature dimension: {node_feature_dim}")

    # Create data loaders
    train_dataset = ShortestPathDataset(train_pairs)
    val_dataset = ShortestPathDataset(val_pairs)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )

    # Initialize model
    model = create_model(
        model_type=args.model_type,
        node_feature_dim=node_feature_dim,
        edge_feature_dim=1,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        output_dim=1,
        dropout=0.1,
        num_heads=args.num_heads,
        use_layer_norm=args.use_layer_norm,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    wandb.config.update(
        {
            "model/num_parameters": num_params,
            "model/type": args.model_type,
            "model/hidden_dim": args.hidden_dim,
            "model/num_layers": args.num_layers,
            "model/num_heads": args.num_heads if args.model_type == "gat" else None,
            "model/use_layer_norm": args.use_layer_norm if args.model_type == "gat" else None,
            "training/effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
            "training/gradient_accumulation_steps": args.gradient_accumulation_steps,
            "training/num_workers": args.num_workers,
            "training/use_amp": args.use_amp,
            "training/lr_scheduler": args.lr_scheduler,
        }
    )

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Mixed precision scaler
    scaler = None
    if args.use_amp and device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()
        print("Using automatic mixed precision (FP16) training")
    elif args.use_amp and device.type != "cuda":
        print("Warning: Mixed precision only supported on CUDA, disabling...")
        args.use_amp = False
    
    # Learning rate scheduler
    scheduler = None
    if args.lr_scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
        )
        print(f"Using CosineAnnealingLR scheduler")
    elif args.lr_scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )
        print(f"Using ReduceLROnPlateau scheduler")

    # Training loop with early stopping
    best_val_mae = float("inf")
    best_epoch_num = 0
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_maes = []

    print("\nStarting training...")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            args.gradient_accumulation_steps,
            args.use_amp,
            scaler,
        )
        train_losses.append(train_loss)

        # Validate
        val_loss, val_mae = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_maes.append(val_mae)
        
        # Update learning rate scheduler
        if scheduler is not None:
            if args.lr_scheduler == "cosine":
                scheduler.step()
            elif args.lr_scheduler == "plateau":
                scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]
            wandb.log({"learning_rate": current_lr})

        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")
        print(f"  Val MAE: {val_mae:.6f}")

        # Log sequential metrics to wandb (per-epoch data)
        # Using metric namespacing to group train_loss and val_loss together
        wandb.log(
            {
                "epoch": epoch + 1,
                "loss/train": train_loss,  # Grouped under "loss"
                "loss/val": val_loss,  # Grouped under "loss"
                "metrics/val_mae": val_mae,  # Separate grouping for MAE
            }
        )

        # Checkpoint best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch_num = epoch + 1
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
            checkpoint_path = checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"  ✓ Saved best model (Val MAE: {val_mae:.6f})")

            # Log best model checkpoint as sequential data (tracks improvement over epochs)
            wandb.log(
                {
                    "best/metrics/val_mae": val_mae,  # Best metrics grouped together
                    "best/epoch": best_epoch_num,  # Best epoch tracking
                }
            )
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

    # Create training curves plots
    try:
        import matplotlib.pyplot as plt

        epochs_range = range(1, len(train_losses) + 1)

        # Dedicated Train vs Val Loss curve
        fig_loss, ax_loss = plt.subplots(figsize=(10, 6))
        ax_loss.plot(
            epochs_range,
            train_losses,
            label="Train Loss",
            marker="o",
            linewidth=2,
            markersize=4,
        )
        ax_loss.plot(
            epochs_range,
            val_losses,
            label="Val Loss",
            marker="s",
            linewidth=2,
            markersize=4,
        )
        ax_loss.set_xlabel("Epoch", fontsize=12)
        ax_loss.set_ylabel("Loss", fontsize=12)
        ax_loss.set_title("Train vs Validation Loss", fontsize=14, fontweight="bold")
        ax_loss.legend(fontsize=11)
        ax_loss.grid(True, alpha=0.3)
        plt.tight_layout()

        # Log the loss curve separately
        wandb.log({"loss_epoch_curve": wandb.Image(fig_loss)})
        plt.close(fig_loss)

        # Additional training curves plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Training/Validation Loss
        axes[0].plot(epochs_range, train_losses, label="Train Loss", marker="o")
        axes[0].plot(epochs_range, val_losses, label="Val Loss", marker="s")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training and Validation Loss")
        axes[0].legend()
        axes[0].grid(True)

        # Validation MAE
        axes[1].plot(epochs_range, val_maes, label="Val MAE", marker="o", color="green")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("MAE")
        axes[1].set_title("Validation MAE")
        axes[1].legend()
        axes[1].grid(True)

        # Combined view
        ax2_twin = axes[2].twinx()
        line1 = axes[2].plot(
            epochs_range, train_losses, label="Train Loss", marker="o", color="blue"
        )
        line2 = axes[2].plot(
            epochs_range, val_losses, label="Val Loss", marker="s", color="red"
        )
        line3 = ax2_twin.plot(
            epochs_range, val_maes, label="Val MAE", marker="^", color="green"
        )
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Loss", color="black")
        ax2_twin.set_ylabel("MAE", color="green")
        axes[2].set_title("Training Overview")
        axes[2].grid(True)

        # Combined legend
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        axes[2].legend(lines, labels, loc="upper left")

        plt.tight_layout()

        # Log the combined plot to wandb
        wandb.log({"training_curves": wandb.Image(fig)})
        plt.close(fig)
    except ImportError:
        print("Warning: matplotlib not available, skipping training curves plot")

    # Log final summary to wandb (single values) - organized by category
    wandb.summary.update(
        {
            # Best metrics (overall best performance)
            "best/metrics/val_mae": best_val_mae,
            "best/epoch": best_epoch_num,
            # Training progress
            "training/total_epochs": len(train_losses),
            "training/completed": True,
            # Final metrics (last epoch performance)
            "loss/final_train": train_losses[-1] if train_losses else None,
            "loss/final_val": val_losses[-1] if val_losses else None,
            "metrics/final_val_mae": val_maes[-1] if val_maes else None,
        }
    )

    # Save artifacts: training history and model checkpoint
    history_file = checkpoint_dir / "training_history.json"
    if history_file.exists():
        artifact = wandb.Artifact("training_history", type="training_data")
        artifact.add_file(str(history_file))
        wandb.log_artifact(artifact)

    checkpoint_path = checkpoint_dir / "best_model.pt"
    if checkpoint_path.exists():
        artifact = wandb.Artifact("best_model", type="model")
        artifact.add_file(str(checkpoint_path))
        wandb.log_artifact(artifact)

    wandb.finish()


if __name__ == "__main__":
    main()
