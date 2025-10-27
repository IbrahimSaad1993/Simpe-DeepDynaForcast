# Graph Neural Network for Tree Classification (Pure PyTorch)

A clean, **pure PyTorch** implementation of graph neural networks for classifying **tree-structured** data — no external graph libraries required.

## Key Features

- ✅ **Pure PyTorch Implementation** — No DGL, no PyTorch Geometric, just PyTorch.
- ✅ **Single GPU Support** — Simple, efficient, and easy to run anywhere.
- ✅ **Clean Code Structure** — Readable, modular files.
- ✅ **Multiple GNN Architectures** — GCN, GAT, GIN, and an LSTM-based message passing model.
- ✅ **Custom Graph Ops** — Scatter/indexing only; all layers written from scratch.

---

## Installation

```bash
# 1) Install PyTorch for your platform (see https://pytorch.org)
pip install torch torchvision torchaudio

# 2) Install the project dependencies (no graph libs needed!)
pip install -r requirements.txt
