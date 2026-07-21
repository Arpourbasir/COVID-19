import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

# Load and preprocess dataset
data = pd.read_csv('Covid Data.csv')
data = data.drop(columns=['DATE_DIED', 'ICU', 'INTUBED'])
data = data.replace({99: np.nan, 97: np.nan, 98: np.nan})
data['PREGNANT'] = data['PREGNANT'].replace({np.nan: 0})
data['CLASIFFICATION_FINAL'] = data['CLASIFFICATION_FINAL'].replace({1: 1, 2: 1, 3: 1, 4: 0, 5: 0, 6: 0, 7: 0})
data = data.dropna()

# Separate features and labels
X = data.drop(columns=['CLASIFFICATION_FINAL'])
y = data['CLASIFFICATION_FINAL']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to numpy arrays for manual implementation
X_train_np = X_train.values
y_train_np = y_train.values
X_test_np = X_test.values
y_test_np = y_test.values

# Custom Decision Tree Class
class CustomDecisionTree:
    def init(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        unique_classes = np.unique(y)

        if len(unique_classes) == 1:
            return unique_classes[0]

        if self.max_depth is not None and depth >= self.max_depth:
            return self._most_common_label(y)

        best_feature, best_threshold = self._find_best_split(X, y)

        if best_feature is None:
            return self._most_common_label(y)

        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold

        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return (best_feature, best_threshold, left_subtree, right_subtree)

    def _find_best_split(self, X, y):
        best_feature, best_threshold, max_gain = None, None, -np.inf

        for feature_idx in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_indices = X[:, feature_idx] <= threshold
                right_indices = X[:, feature_idx] > threshold

                if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
                    continue

                gain = self._information_gain(y, y[left_indices], y[right_indices])
                if gain > max_gain:
                    best_feature, best_threshold, max_gain = feature_idx, threshold, gain

        return best_feature, best_threshold

    def _information_gain(self, parent, left, right):
        return self._entropy(parent) - (
            len(left) / len(parent) * self._entropy(left) +
            len(right) / len(parent) * self._entropy(right)
        )

    def _entropy(self, labels):
        _, counts = np.unique(labels, return_counts=True)
        probabilities = counts / len(labels)
        return -np.sum(probabilities * np.log2(probabilities + 1e-9))

    def _most_common_label(self, labels):
        return np.bincount(labels).argmax()

    def predict(self, X):
        return np.array([self._traverse_tree(sample, self.root) for sample in X])

    def _traverse_tree(self, sample, tree):
        if not isinstance(tree, tuple):
            return tree

        feature_idx, threshold, left, right = tree
        if sample[feature_idx] <= threshold:
            return self._traverse_tree(sample, left)
        else:
            return self._traverse_tree(sample, right)

# Initialize KFold and models
kf = KFold(n_splits=3, shuffle=True, random_state=42)
custom_f1_scores = []
sklearn_f1_scores = []

custom_tree = CustomDecisionTree(max_depth=3)
sklearn_tree = DecisionTreeClassifier(random_state=42, max_depth=3)
# Perform KFold cross-validation
for train_idx, val_idx in kf.split(X_train_np):
    X_train_fold, X_val_fold = X_train_np[train_idx], X_train_np[val_idx]
    y_train_fold, y_val_fold = y_train_np[train_idx], y_train_np[val_idx]

    custom_tree.fit(X_train_fold, y_train_fold)
    sklearn_tree.fit(X_train_fold, y_train_fold)

    custom_preds = custom_tree.predict(X_val_fold)
    sklearn_preds = sklearn_tree.predict(X_val_fold)

    custom_f1_scores.append(f1_score(y_val_fold, custom_preds, average='weighted'))
    sklearn_f1_scores.append(f1_score(y_val_fold, sklearn_preds, average='weighted'))

# Calculate and print average F1 scores
average_custom_f1 = np.mean(custom_f1_scores)
average_sklearn_f1 = np.mean(sklearn_f1_scores)

print("Custom Decision Tree F1-Score:", average_custom_f1)
print("Sklearn Decision Tree F1-Score:", average_sklearn_f1)
