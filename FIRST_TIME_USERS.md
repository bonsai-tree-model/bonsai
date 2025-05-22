# Getting Started with BONSAI

This guide provides a quick orientation for first time users of the BONSAI repository.
It walks through installation, running the bundled tests, and locating example datasets.

## 1. Requirements
- Python 3.8 or later
- `pandas` and `scikit-learn` for running the included experiment scripts

You can install the required Python packages with:

```bash
pip install pandas scikit-learn
```

## 2. Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd bonsai
   ```
2. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. Install the package and its dependencies:
   ```bash
   pip install -e .
   ```

## 3. Running Tests
The project includes a small test suite. To execute it, run:

```bash
python pytest
```

All tests should report `PASSED` when completed.

## 4. Example Datasets
Sample datasets used by the experiments are located under `Experiments/dataset/`.
Each dataset directory contains train and test CSV files. For instance, the Iris
dataset files reside in `Experiments/dataset/Iris/`.

## 5. Example Scripts
Several example scripts demonstrating decision tree training can be found in the
`Experiments/` directory. These scripts load a dataset, fit a model, and report
metrics. To try one out, run:

```bash
python Experiments/cart.py
```

Feel free to edit the dataset paths at the top of the script to experiment with
other datasets.
