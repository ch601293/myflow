# Flow Matching with Gaussian Process Priors for Probabilistic Time Series Forecasting

Marcel Kollovieh, Marten Lienen, David Lüdke, Leo Schwinn, Stephan Günnemann


## Installation

If you want to run our code, start by setting up the python environment.

### Cross-Platform Installation

This project now supports **Linux**, **Windows**, and **macOS** (both Intel and Apple Silicon).

#### Using pixi (Recommended)

We use [pixi](https://pixi.sh/) to easily set up reproducible environments based on conda packages.

**On Linux/macOS:**
```sh
# Install pixi
curl -fsSL https://pixi.sh/install.sh | bash

# Clone the repository
git clone https://github.com/marcelkollovieh/TSFlow.git

# Change into the repository
cd TSFlow

# Install and activate the environment
pixi shell
```

**On Windows:**
```powershell
# Install pixi (using PowerShell)
iwr -useb https://pixi.sh/install.ps1 | iex

# Clone the repository
git clone https://github.com/marcelkollovieh/TSFlow.git

# Change into the repository
cd TSFlow

# Install and activate the environment
pixi shell
```

#### Using pip (Alternative)

Alternatively, you can use pip to install dependencies:

```sh
pip install -r requirements.txt
pip install -e .
```

**Optional Performance Enhancement (Linux/macOS only):**
```sh
# Install pykeops for GPU-accelerated kernel operations
# Note: Not recommended on Windows due to complex build requirements
pip install pykeops>=2.1.1
```

### Platform-Specific Notes

- **pykeops**: This package provides GPU-accelerated operations but has complex build requirements on Windows (requires C++ compiler and CUDA). The code automatically falls back to pure PyTorch implementations if pykeops is not available, with minimal performance impact for most use cases.

- **POT (Python Optimal Transport)**: Fully cross-platform compatible and works on all supported platforms.

## Training

Start a training by running `train.py` with the your settings, for example:
```sh
python bin/train_model.py -c configs_local/train_conditional.yaml
```

The results will be logged in `./logs`

### Monitoring Training with TensorBoard

This project uses **TensorBoard** for experiment tracking and visualization (cross-platform compatible). To monitor your training:

```sh
# Start TensorBoard (in a separate terminal)
tensorboard --logdir=./logs

# Then open your browser to http://localhost:6006
```

TensorBoard will show:
- **Scalars**: Training loss, validation metrics (CRPS, ND, NRMSE)
- **Images**: Forecast visualization samples
- **HParams**: Hyperparameter configurations

**Note**: We replaced the previous `aim` package with `tensorboard` for better cross-platform compatibility (Windows, macOS, Linux).

## Citation

If you build upon this work, please cite our paper as follows.

```bibtex
@article{kollovieh2024flow,
  title = {Flow Matching with Gaussian Process Priors for Probabilistic Time Series Forecasting},
  author = {Kollovieh, Marcel and Lienen, Marten and L{\"u}dke, David and Schwinn, Leo and G{\"u}nnemann, Stephan},
  journal = {The Thirteenth International Conference on Learning Representations},
  shortjournal = {ICLR},
  year = {2025},
}
```