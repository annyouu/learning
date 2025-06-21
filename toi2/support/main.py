import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Irisデータセットの読み込み
iris = load_iris()
x = iris.data
y = iris.target

# データの分割/訓練
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# 特徴量のスケーリング
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

print("特徴量のスケーリングが完了しました。\n")

# 学習と予測の実施
clf = SVC(kernel='rbf', C=1.0, random_state=42)
clf.fit(x_train_scaled, y_train)

print("サポートベクターマシンモデルの学習が終わった\n")

y_predicted = clf.predict(x_test_scaled)

# 精度 (Accuracy)
accuracy = accuracy_score(y_test, y_predicted)
print(f"精度: {accuracy:.4f}") # 小数点以下4桁まで表示

# 適合率 (Precision)
# 多クラス分類のため average='macro' を使用 (各クラスの適合率を単純平均)
precision = precision_score(y_test, y_predicted, average='macro')
print(f"適合率: {precision:.4f}")

# 再現率 (Recall)
# 多クラス分類のため average='macro' を使用 (各クラスの再現率を単純平均)
recall = recall_score(y_test, y_predicted, average='macro')
print(f"再現率: {recall:.4f}")

# F1 スコア (F1-score)
# 多クラス分類のため average='macro' を使用
f1 = f1_score(y_test, y_predicted, average='macro')
print(f"F1 スコア: {f1:.4f}\n")