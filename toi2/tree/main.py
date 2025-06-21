import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Irisデータセットの読み込み
data = load_iris()
x = data.data
y = data.target

# データの分割/訓練
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

# 学習と予測の実施
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(x_train, y_train)

print("決定木モデルの学習が終わった\n")

y_predicted = clf.predict(x_test)


# 精度
# accuracy = accuracy_score(y_test, y_predicted)
# print(f"精度: {accuracy:.4f}")
print(f"精度: {sum(y_predicted == y_test) / len(y_test): .4f}")

# 適合率
precision = precision_score(y_test, y_predicted, average='macro')
print(f"適合率: {precision:.4f}")

# 再現性
recall = recall_score(y_test, y_predicted, average='macro')
print(f"再現率: {recall:.4f}")

# F1
f1 = f1_score(y_test, y_predicted, average='macro')
print(f"F1 スコア: {f1:.4f}\n")

