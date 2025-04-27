# がんの診断を行う場合、
# 患者の年齢や血圧、診断結果などの特徴量を使って、がんかどうかを予測する。
# ロジスティック回帰では、これらの特徴量に基づいて確率を出力し0.5を境にがんかどうかを確率て判定する。

from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.datasets import load_breast_cancer # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import accuracy_score # type: ignore

# 乳がんのデータセットを読み込み
data = load_breast_cancer()
X = data.data # 特徴量
Y = data.target # ラベル(0 ok, 1 NG)

# 訓練データとテストデータに分ける(80% 訓練, 20% テスト)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# ロジステック回帰モデルを訓練
model = LogisticRegression(max_iter=10000) # 最大反復回数を増やすことで収束しやすくする
model.fit(X_train, Y_train)

# テストデータで予測
Y_pred = model.predict(X_test)

# 精度を評価
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuracy: {accuracy}")
