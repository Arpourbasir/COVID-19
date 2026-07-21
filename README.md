# 🌲 COVID-19 Decision Tree Classifier  
### Custom Implementation vs. Scikit-Learn

A machine learning project that implements a **Decision Tree Classifier from scratch** using **NumPy** and **Python**, then compares its performance with **Scikit-Learn's DecisionTreeClassifier** on a real-world **COVID-19** dataset.

This project was developed for the **Introduction to Data Mining** course at **K. N. Toosi University of Technology (KNTU)**.

---

# 📖 Project Overview

The primary objective of this project is to implement the **Decision Tree learning algorithm** from first principles without using machine learning libraries for tree construction.

The custom implementation is evaluated against **Scikit-Learn's DecisionTreeClassifier** using **3-Fold Cross-Validation** and the **Weighted F1-Score** metric.

The project demonstrates both the theoretical understanding of decision trees and their practical application to a real-world medical dataset.

---

# ✨ Features

- 🌲 Decision Tree implementation from scratch
- 📊 Entropy & Information Gain based splitting
- 🧮 Recursive tree construction
- 🔍 Automatic threshold selection
- 📈 Weighted F1-Score evaluation
- 🔄 3-Fold Cross-Validation
- ⚖️ Performance comparison with Scikit-Learn
- 🧹 Data preprocessing pipeline
- 📉 Configurable maximum tree depth

---

# 📂 Project Structure

```text
COVID-Decision-Tree/
│
├── DecisionTree.ipynb          # Main implementation
├── Covid Data.csv              # Dataset
├── README.md                   # Project documentation
└── requirements.txt            # Dependencies (optional)
```

---

# 📊 Dataset

The project uses the **COVID-19 Dataset** from **Kaggle** containing patient demographic and clinical information.

The target variable is derived from the **CLASIFFICATION_FINAL** column.

---

# 🧹 Data Preprocessing

Several preprocessing steps are applied before training.

## Feature Removal

The following columns are removed because they contain excessive missing values or are not used in the model:

- DATE_DIED
- ICU
- INTUBED

---

## Missing Value Handling

Special values representing unknown information are replaced with `NaN`.

```python
97 → NaN
98 → NaN
99 → NaN
```

Rows containing missing values are removed.

For the **PREGNANT** feature:

```python
NaN → 0
```

---

## Target Variable Transformation

The original multiclass labels are converted into a binary classification problem.

| Original Label | New Label |
|---------------|-----------|
| 1 | Positive (1) |
| 2 | Positive (1) |
| 3 | Positive (1) |
| 4 | Negative (0) |
| 5 | Negative (0) |
| 6 | Negative (0) |
| 7 | Negative (0) |

---

# 🌲 Custom Decision Tree Algorithm

The project implements the complete Decision Tree algorithm from scratch using only **NumPy**.

The implementation includes:

- Entropy calculation
- Information Gain computation
- Best split search
- Recursive tree construction
- Prediction traversal
- Leaf node creation
- Maximum depth stopping criterion

---

# 🧠 Decision Tree Workflow

```text
Training Data
      │
      ▼
Entropy Calculation
      │
      ▼
Find Best Feature & Threshold
      │
      ▼
Information Gain
      │
      ▼
Recursive Split
      │
      ▼
Leaf Nodes
      │
      ▼
Prediction
```

---

# 📐 Mathematical Foundation

## Entropy

\[
H(S)= -\sum_i p_i\log_2(p_i)
\]

Entropy measures the impurity of a dataset.

---

## Information Gain

\[
IG(S,A)=H(S)-\sum_{v}
\frac{|S_v|}{|S|}
H(S_v)
\]

The split with the highest Information Gain is selected at each node.

---

# ⚙️ Model Training

The project compares two classifiers:

### Custom Implementation

- Built entirely from scratch
- Uses recursive binary splitting
- Entropy criterion
- Information Gain optimization

### Scikit-Learn Implementation

```python
DecisionTreeClassifier(max_depth=3)
```

Both models use identical training data and hyperparameters for a fair comparison.

---

# 📈 Model Evaluation

Performance is evaluated using:

- **3-Fold Cross-Validation**
- **Weighted F1-Score**

```python
kf = KFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)
```

Evaluation metric:

```python
f1_score(
    y_true,
    predictions,
    average="weighted"
)
```

---

# 🚀 Getting Started

## Requirements

- Python 3.9+
- Jupyter Notebook

---

## Required Libraries

```text
numpy
pandas
scikit-learn
matplotlib (optional)
seaborn (optional)
```

Install dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

---

# ▶️ Running the Project

Clone the repository:

```bash
git clone https://github.com/your-username/COVID-Decision-Tree.git
```

Navigate to the project directory:

```bash
cd COVID-Decision-Tree
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```
DecisionTree.ipynb
```

Run all notebook cells.

---

# 🛠 Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| NumPy | Numerical Computing |
| Pandas | Data Processing |
| Scikit-Learn | Model Comparison |
| Jupyter Notebook | Development Environment |
| Kaggle Dataset | COVID-19 Dataset |

---

# 🎯 Learning Outcomes

This project demonstrates practical implementation of:

- Decision Trees
- Recursive Algorithms
- Information Theory
- Entropy
- Information Gain
- Binary Classification
- Model Evaluation
- Cross Validation
- Machine Learning Fundamentals

---

# 🔮 Future Improvements

Possible future enhancements include:

- 🌳 Random Forest implementation
- ⚡ Gradient Boosting comparison
- 📊 Decision tree visualization
- 🔍 Feature importance analysis
- ✂️ Cost-complexity pruning
- 🎯 Hyperparameter optimization
- 📈 ROC Curve & AUC evaluation
- 🧪 Unit testing
- 🚀 Parallelized training

---

# 📚 Academic Information

**Course**

Introduction to Data Mining

**University**

K. N. Toosi University of Technology (KNTU)

**Semester**

Fall 2024

**Dataset**

COVID-19 Dataset (Kaggle)

---

# 🤝 Contributing

Contributions, bug reports, and feature requests are welcome.

Feel free to fork the repository and submit a Pull Request.

---

# 📄 License

This project was developed for educational purposes as part of a university assignment.

You are free to study, modify, and extend the implementation for learning and research purposes.

---

# 👨‍💻 Arpourbasir

**Alireza Pourbasir**

GitHub: https://github.com/Arpourbasir

---

⭐ If you found this project useful, consider giving it a **Star ⭐** on GitHub!
