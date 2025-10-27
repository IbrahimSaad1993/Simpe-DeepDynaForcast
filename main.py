"""Main entry point for training and evaluation."""

import os
import logging
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import get_arguments
from dataset import TreeGraphDataset, collate_graphs
from models import create_model
from trainer import Trainer


def setup_logging(log_level):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    """Main function."""
    # Parse arguments
    args = get_arguments()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Set random seed
    set_seed(args.seed)
    logger.info(f"Using seed: {args.seed}")
    
    # Create save directory
    save_dir = f"experiments/{args.model}_{args.model_num}"
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Save directory: {save_dir}")
    
    # Check device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = TreeGraphDataset(args, phase='train')
    val_dataset = TreeGraphDataset(args, phase='valid')
    test_dataset = TreeGraphDataset(args, phase='test')
    
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create data loaders with custom collate function
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        collate_fn=collate_graphs
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        collate_fn=collate_graphs
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        collate_fn=collate_graphs
    )
    
    # Create model
    logger.info(f"Creating {args.model.upper()} model...")
    model = create_model(args)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, test_loader, args, save_dir, args.checkpoint)
    
    # Train or evaluate
    if args.mode == 'train':
        logger.info("Starting training...")
        trainer.train()
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_metrics = trainer.test()
        
    elif args.mode == 'eval':
        logger.info("Evaluating on test set...")
        test_metrics = trainer.test()
    
    logger.info("Done!")


if __name__ == "__main__":
    main()