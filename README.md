# Simple Implementation of DeepDynaForecast

A clean implementation of DeepDynaForecast using **pure PyTorch** (no external graph libraries!).

## Key Features

✅ **Pure PyTorch Implementation** - No DGL, no PyTorch Geometric, just PyTorch!  
✅ **Single GPU Support** - Simple and efficient  
✅ **Clean Code Structure** - Easy to read and modify  
✅ **Multiple GNN Architectures** - GCN, GAT, GIN, LSTM-based models  
✅ **Custom Graph Operations** - All graph convolutions implemented from scratch  

---

## Installation

```bash
# Install dependencies (NO graph libraries needed!)
pip install -r requirements.txt
```

---

## Available Models

All models are implemented in pure PyTorch:

- **GCN** - Graph Convolutional Network
- **GAT** - Graph Attention Network
- **GIN** - Graph Isomorphism Network
- **LSTM** - Long Short-Term Memory based model

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

## Dataset Setup

For training the model, you need to download and preprocess the dataset from the original [DeepDynaForecast GitHub repository](https://github.com/lab-smile/DeepDynaForecast/tree/main). 

### Steps:
1. Download the dataset from the original repository
2. Preprocess the data as needed
3. Place the datasets into the appropriate folder

**Example:** If the dataset is for resp + TB, provide the full path for cleaned_data during training.

---

## Usage

### Training

```bash
python main.py --mode train \
               --ds_dir 'path_for_cleaned_data' \
               --ds_name 'ddf_resp+TB_20230222' \
               --model gcn \
               --hidden_dim 128 \
               --num_layers 20 \
               --batch_size 4 \
               --lr 0.001 \
               --max_epochs 100
```

### Evaluation

For evaluation on already saved models:

1. Download the checkpoints from the original [DeepDynaForecast GitHub repository](https://github.com/lab-smile/DeepDynaForecast/tree/main)
2. Place these models in the appropriate folder
3. Run the evaluation command:

```bash
python main.py --mode eval \
               --checkpoint 'path_for_model.pth.tar_in_ddf_2_folder' \
               --model gcn
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

### Evaluation
- `--checkpoint`: Path to model checkpoint file

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

## Acknowledgments

This implementation uses datasets from [DeepDynaForecast](https://github.com/lab-smile/DeepDynaForecast/tree/main). Please cite their work if you use their datasets.
