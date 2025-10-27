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
# Install PyTorch (check https://pytorch.org for your system)
pip install torch torchvision torchaudio

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

## How It Works

### Graph Representation
Each graph is represented using standard PyTorch tensors:
- `x`: Node features [num_nodes, feature_dim]
- `edge_index`: Edge connectivity [2, num_edges] 
- `edge_attr`: Edge features [num_edges, edge_dim]
- `y`: Node labels [num_nodes]

### Batching
Multiple graphs are batched into a single disconnected graph using the `collate_graphs` function. Node indices are offset appropriately.

### Graph Operations
All graph operations (message passing, aggregation, attention) are implemented using PyTorch's scatter operations:
- `scatter_add`: For aggregating neighbor messages
- `index_select`: For gathering neighbor features
- No external graph library needed!

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
