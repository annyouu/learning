import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # SVM用
from sklearn.svm import SVC # SVMモデル
from sklearn.tree import DecisionTreeClassifier # 決定木モデル
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# --- 共通のデータセット読み込みと分割 ---
iris = load_iris()
x = iris.data  # 特徴量データ
y = iris.target # 正解ラベル

# データの分割/訓練 (8:2)
# random_state を固定し、どちらのモデルでも同じデータを使うようにする
# stratify=y でクラスの比率も訓練/テストで均等にする (推奨)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

print(f"訓練データのサンプル数: {len(x_train)}")
print(f"テストデータのサンプル数: {len(x_test)}\n")

# --- サポートベクターマシン (SVM) モデル ---
print("======================================")
print("     サポートベクターマシン (SVM)     ")
print("======================================")

# 1. 特徴量のスケーリング (SVMには必須の前処理)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

print("SVM用：特徴量のスケーリングが完了しました。\n")

# 2. 学習と予測の実施 (SVC)
svm_clf = SVC(kernel='rbf', C=1.0, random_state=42)
svm_clf.fit(x_train_scaled, y_train) # スケーリングされた訓練データを使用

print("SVMモデルの学習が終わった\n")

# 予測の実施
y_pred_svm = svm_clf.predict(x_test_scaled) # スケーリングされたテストデータを使用

# 3. 評価指標の表示
print("--- SVMモデル評価指標 ---")
print(f"精度 (Accuracy): {accuracy_score(y_test, y_pred_svm):.4f}")
print(f"適合率 (Precision - Macro Avg): {precision_score(y_test, y_pred_svm, average='macro'):.4f}")
print(f"再現率 (Recall - Macro Avg): {recall_score(y_test, y_pred_svm, average='macro'):.4f}")
print(f"F1 スコア (F1-score - Macro Avg): {f1_score(y_test, y_pred_svm, average='macro'):.4f}\n")

# 4. 混同行列の可視化 (SVM)
cm_svm = confusion_matrix(y_test, y_pred_svm)
disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=iris.target_names)

fig_svm, ax_svm = plt.subplots(figsize=(7, 7))
disp_svm.plot(cmap=plt.cm.Blues, ax=ax_svm)
ax_svm.set_title('Confusion Matrix for Iris Dataset (SVM)')
plt.show()


# --- 決定木 (Decision Tree) モデル ---
print("\n======================================")
print("        決定木 (Decision Tree)        ")
print("======================================")

# 1. 学習と予測の実施 (Decision Tree)
# 決定木はスケーリング不要なので、元の x_train を使用
dt_clf = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_clf.fit(x_train, y_train) # スケーリングされていない訓練データを使用

print("決定木モデルの学習が終わった\n")

# 予測の実施
y_pred_dt = dt_clf.predict(x_test) # スケーリングされていないテストデータを使用

# 2. 評価指標の表示
print("--- 決定木モデル評価指標 ---")
# あなたの元のコードに合わせて sum()/len() を使用しますが、accuracy_score() を使う方が一般的です
print(f"精度 (Accuracy): {sum(y_pred_dt == y_test) / len(y_test):.4f}")
print(f"適合率 (Precision - Macro Avg): {precision_score(y_test, y_pred_dt, average='macro'):.4f}")
print(f"再現率 (Recall - Macro Avg): {recall_score(y_test, y_pred_dt, average='macro'):.4f}")
print(f"F1 スコア (F1-score - Macro Avg): {f1_score(y_test, y_pred_dt, average='macro'):.4f}\n")

# 3. 混同行列の可視化 (Decision Tree)
cm_dt = confusion_matrix(y_test, y_pred_dt)
disp_dt = ConfusionMatrixDisplay(confusion_matrix=cm_dt, display_labels=iris.target_names)

fig_dt, ax_dt = plt.subplots(figsize=(7, 7))
disp_dt.plot(cmap=plt.cm.Blues, ax=ax_dt)
ax_dt.set_title('Confusion Matrix for Iris Dataset (Decision Tree)')
plt.show()

