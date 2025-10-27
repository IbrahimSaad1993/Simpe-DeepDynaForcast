# Graph Neural Network for Tree Classification

A clean implementation of graph neural networks for tree-structured data classification using **pure PyTorch** (no external graph libraries!).

## Key Features

✅ **Pure PyTorch Implementation** - No DGL, no PyTorch Geometric, just PyTorch!  
✅ **Single GPU Support** - Simple and efficient  
✅ **Clean Code Structure** - Easy to read and modify  
✅ **Multiple GNN Architectures** - GCN, GAT, GIN, LSTM-based models  
✅ **Custom Graph Operations** - All graph convolutions implemented from scratch  

---

## Installation

```bash

# Install other dependencies (NO graph libraries needed!)
pip install -r requirements.txt
```

---

## Usage

### Training

```bash
python main.py --mode train \
               --model gcn \
               --hidden_dim 128 \
               --num_layers 20 \
               --batch_size 4 \
               --lr 0.001 \
               --max_epochs 100
```

### Evaluation

```bash
python main.py --mode eval \
               --model gcn \
               --restore
```

---

## Available Models

All models are implemented in pure PyTorch:

- **GCN**
- **GAT** 
- **GIN**
- **LSTM**

---

## Project Structure

```
.
├── config.py           # Configuration parameters
├── dataset.py          # Dataset loading (pure PyTorch)
├── models.py           # GNN models (pure PyTorch)
├── trainer.py          # Training/evaluation logic
├── utils.py            # Metrics and visualization
├── main.py             # Entry point
├── requirements.txt    # Dependencies (no graph libraries!)
└── README.md           # Documentation
```

---


## Configuration Options

### General
- `--seed`: Random seed (default: 123)
- `--device`: Device to use (cuda/cpu)
- `--mode`: train or eval

### Data
- `--ds_name`: Dataset name
- `--ds_dir`: Dataset directory
- `--batch_size`: Batch size (default: 4)
- `--num_workers`: DataLoader workers

### Model
- `--model`: Model type (gcn/gat/gin/lstm)
- `--hidden_dim`: Hidden dimension (default: 128)
- `--num_layers`: Number of layers (default: 20)
- `--dropout`: Dropout rate (default: 0.5)

### Training
- `--max_epochs`: Maximum epochs (default: 100)
- `--optimizer`: Optimizer (Adam/SGD/RMSprop)
- `--lr`: Learning rate (default: 0.001)
- `--weight_decay`: Weight decay (default: 0.0004)
- `--early_stopping`: Early stopping patience (default: 10)

---

## License

MIT — see `LICENSE`.

## Citation

```bibtex
@software{gnn_tree_pytorch_2025,
  title  = {Graph Neural Network for Tree Classification (Pure PyTorch)},
  author = {Your Name},
  year   = {2025},
  url    = {https://github.com/your/repo}
}
```
