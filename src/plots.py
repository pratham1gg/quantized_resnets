import matplotlib.pyplot as plt
import os
from pathlib import Path
from typing import Optional

from metrics import MetricsTracker


def plot_accuracy(tracker: MetricsTracker, save_path: Optional[str] = None):
    """
    Plot top-1 and top-5 accuracy over batches.
    """
    if not tracker.top1_running:
        print("No accuracy data to plot")
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(tracker.top1_running, label='Top-1 Accuracy', linewidth=2)
    plt.plot(tracker.top5_running, label='Top-5 Accuracy', linewidth=2)
    plt.xlabel('Batch')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved accuracy plot to {save_path}")
    
    # Only show if interactive
    if plt.get_backend() != 'agg':
        plt.show()
    plt.close()


def plot_timing(tracker: MetricsTracker, save_path: Optional[str] = None):
    """
    Plot batch time and inference time over batches.
    """
    if not tracker.batch_times_s:
        print("No timing data to plot")
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot([t * 1000 for t in tracker.batch_times_s], label='Batch Time', linewidth=2)
    plt.plot([t * 1000 for t in tracker.infer_times_s], label='Inference Time', linewidth=2)
    plt.xlabel('Batch')
    plt.ylabel('Time (ms)')
    plt.title('Batch and Inference Timing')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved timing plot to {save_path}")

    if plt.get_backend() != 'agg':
        plt.show()
    plt.close()


def plot_loss(tracker: MetricsTracker, save_path: Optional[str] = None):
    """
    Plot loss over batches.
    """
    if not tracker.losses:
        print("No loss data to plot")
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(tracker.losses, label='Loss', linewidth=2, color='red')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved loss plot to {save_path}")

    if plt.get_backend() != 'agg':
        plt.show()
    plt.close()


def plot_all(tracker: MetricsTracker, save_dir: Optional[str] = None):
    """
    Plot all metrics in a 3-subplot figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Accuracy
    if tracker.top1_running:
        axes[0].plot(tracker.top1_running, label='Top-1', linewidth=2)
        axes[0].plot(tracker.top5_running, label='Top-5', linewidth=2)
        axes[0].set_xlabel('Batch')
        axes[0].set_ylabel('Accuracy (%)')
        axes[0].set_title('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # Timing
    if tracker.batch_times_s:
        axes[1].plot([t * 1000 for t in tracker.batch_times_s], label='Batch', linewidth=2)
        axes[1].plot([t * 1000 for t in tracker.infer_times_s], label='Inference', linewidth=2)
        axes[1].set_xlabel('Batch')
        axes[1].set_ylabel('Time (ms)')
        axes[1].set_title('Timing')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    # Loss
    if tracker.losses:
        axes[2].plot(tracker.losses, linewidth=2, color='red')
        axes[2].set_xlabel('Batch')
        axes[2].set_ylabel('Loss')
        axes[2].set_title('Loss')
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_path = Path(save_dir) / "metrics_summary.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved metrics summary to {save_path}")

    if plt.get_backend() != 'agg':
        plt.show()
    plt.close()