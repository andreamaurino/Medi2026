# Pucktrick: a Strategy Driven Data Corruption Library for Sensitivity Analysis
This repository contains code and results for a systematic evaluation of label noise injection mechanisms (NCAR, NAR, NNAR) using the [PuckTrick] library. We assess how different corruption mechanisms affect three ML models (Random Forest, Linear SVM, K-Nearest Neighbors) across two real-world datasets.

## Quick Start

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd medi-experiment
```

### 2. Install Dependencies

#### Python 3.11+

```bash
python --version  # Should be 3.11 or higher
```

#### Create Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Install Required Packages

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install pucktrick>=1.0.7
pip install scikit-learn>=1.3.2
pip install pandas>=1.5.0
pip install numpy>=1.24.0
pip install scipy>=1.10.0
pip install matplotlib>=3.7.0
pip install seaborn>=0.12.0
```

### 3. Download Datasets

#### Wine Quality Red Dataset

```bash
mkdir -p datasetroot
wget https://archive.ics.uci.edu/ml/machine-learning-databases/wine/winequality-red.csv -O datasetroot/winequality-red.csv
```

Or manually download from: [UCI ML Repository - Wine Quality](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)

#### Pen Digits Dataset

The Pen Digits dataset is downloaded automatically from OpenML:

```python
from sklearn.datasets import fetch_openml
import pandas as pd

pendigits = fetch_openml('pendigits', version=1, parser='auto')
df = pd.DataFrame(pendigits.data)
df['class'] = pendigits.target
df.to_csv('datasetroot/pendigit-multi.csv', index=False)
```

Or run once during the first experiment execution.

## Project Structure

```
medi-experiment/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── medi.py                        # Main experiment script
├── results.csv                    # Experimental results (2,880 runs)
├── datasetroot/                   # Dataset directory
│   ├── winequality-red.csv
│   └── pendigit-multi.csv

## Usage

### Run the Experiment

```bash
python medi.py
```

This will:
- Load the two datasets from `datasetroot/`
- Generate 2,880 experimental configurations (2 datasets × 3 error types × 4 intensities × 3 models × 2 feature modes × 20 trials)
- Apply label noise using PuckTrick (NCAR, NAR, NNAR)
- Inject feature outliers in composed mode
- Train and evaluate models
- Save results to `results.csv`

**Estimated runtime:** 20-30 minutes (depends on hardware)

## Experiment Configuration

Key parameters in `medi.py`:

```python
ERROR_TYPES = ["NCAR", "NAR", "NNAR"]          # Label noise mechanisms
INTENSITIES = [0.10, 0.20, 0.30, 0.50]        # Corruption levels (10%-50%)
MODELS = {
    "RF": RandomForestClassifier(n_estimators=100),
    "LinearSVM": LinearSVC(max_iter=10000, dual='auto'),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}
NUM_TRIALS = 20                                 # Independent train-test splits
```

## Datasets

| Dataset | Instances | Features | Classes | Characteristics |
|---------|-----------|----------|---------|-----------------|
| Wine Quality Red | 1,599 | 11 | 6 | Imbalanced, continuous, physicochemical properties |
| Pen Digits | 10,000 | 16 | 10 | Balanced, high-dimensional, temporal trajectories |

## Results Summary

**Key Findings:**

1. **Non-linear Accuracy Decay**: Models maintain ~73% accuracy at 10% corruption but collapse to 35% at 50%
2. **Dataset Dominance**: Pen Digits (balanced) achieves 74% accuracy vs. Wine Quality (imbalanced) at 45%
3. **Mechanism Equivalence**: NCAR ≈ NAR ≈ NNAR (no significant difference, *p* > 0.05)
4. **Catastrophic Composed Corruption**: Label-only (69%) vs. Label+Feature (49%) — 41% relative decline
5. **Critical Collapse at 50%**: Composed corruption drops from 61% to just 9% at highest intensity

**Publication Outputs:**
- `results.csv` - Full 2,880-row results table


## Requirements

- **Python:** 3.11+
- **Core libraries:** scikit-learn 1.3.2+, pandas 1.5+, numpy 1.24+
- **Error injection:** PuckTrick 1.0.7+
- **Analysis:** scipy 1.10+
- **Visualization:** matplotlib 3.7+, seaborn 0.12+

See `requirements.txt` for exact versions.

## Troubleshooting

### Issue: PuckTrick import fails

```bash
pip install --upgrade pucktrick
```

### Issue: Dataset file not found

Ensure files exist in `datasetroot/`:
```bash
ls datasetroot/
# Should show: pendigit-multi.csv  winequality-red.csv
```

### Issue: Slow execution

- Reduce `NUM_TRIALS` in `medi.py` (default: 20)
- Use smaller intensity values: `[0.1, 0.3, 0.5]`
- Experiment runs on CPU; GPU acceleration not yet supported

### Issue: Memory errors on large experiments

Reduce batch size or run on a machine with 8GB+ RAM.

## License

See LICENSE file for details.

## Contact & Questions

**Author:** Andrea Maurino  
**Institution:** Università degli Studi di Milano-Bicocca  
**Email:** andrea.maurino@unimib.it

For issues, questions, or contributions, please open a GitHub issue.

---

**Last Updated:** June 2026
