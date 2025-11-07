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
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    args = get_arguments()

    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    set_seed(args.seed)
    logger.info(f"Using seed: {args.seed}")

    save_dir = f"experiments/{args.model}_{args.model_num}"
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Save directory: {save_dir}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create model
    logger.info(f"Creating {args.model.upper()} model...")
    model = create_model(args).to(device)

    # Optionally load checkpoint (useful for eval)
    if args.checkpoint and os.path.isfile(args.checkpoint):
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        state = ckpt.get("state_dict") or ckpt.get("model") or ckpt
        model.load_state_dict(state, strict=False)
    elif args.mode == "eval":
        logger.warning("No valid --checkpoint provided; evaluating current weights.")

    # ----- Branch on mode -----
    if args.mode == 'eval':
        # EVAL-ONLY: use test.csv only
        # test_csv = getattr(args, "test_csv", None)
        # if not test_csv:
        #     test_csv = os.path.join(args.ds_dir, "test.csv")
        # if not os.path.isfile(test_csv):
        #     raise FileNotFoundError(f"test.csv not found at: {test_csv}")

        # logger.info(f"Loading test dataset from: {test_csv}")
        test_dataset = TreeGraphDataset(args, phase='test')
        logger.info(f"Test samples: {len(test_dataset)}")

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_graphs,
            pin_memory=torch.cuda.is_available(),
        )

        trainer = Trainer(model, train_loader=None, val_loader=None, test_loader=test_loader,
                          args=args, save_dir=save_dir, checkpoint_path = args.checkpoint)
        logger.info("Evaluating on test set...")
        test_metrics = trainer.test()
        logger.info(f"Test metrics: {test_metrics}")

    else:
        # TRAINING: load train/val/test as usual
        logger.info("Loading datasets (train/val/test)...")
        train_dataset = TreeGraphDataset(args, phase='train')
        val_dataset   = TreeGraphDataset(args, phase='valid')
        test_dataset  = TreeGraphDataset(args, phase='test')
        logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, collate_fn=collate_graphs,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_graphs,
            pin_memory=torch.cuda.is_available(),
        )
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_graphs,
            pin_memory=torch.cuda.is_available(),
        )

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {num_params:,}")

        trainer = Trainer(model, train_loader, val_loader, test_loader,
                          args=args, save_dir=save_dir, checkpoint=args.checkpoint)

        logger.info("Starting training...")
        trainer.train()

        logger.info("Evaluating on test set...")
        test_metrics = trainer.test()
        logger.info(f"Test metrics: {test_metrics}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
