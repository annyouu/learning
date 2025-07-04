import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# CSV読み込み
df = pd.read_csv('cinema.csv')

# データ確認
# print(df.head(3))

# 欠損値を平均で補完
df2 = df.fillna(df.mean())

# 外れ値の削除
no = df2[(df2['SNS2'] > 1000) & (df2['sales'] < 8500)].index
df3 = df2.drop(no, axis=0)

# 特徴量と目的変数
col = ['SNS1', 'SNS2', 'actor', 'original']
x = df3[col]
t = df3['sales']

# データ分割
x_train, x_test, y_train, y_test = train_test_split(x, t, test_size=0.2, random_state=0)

# モデルの作成
model = LinearRegression()
model.fit(x_train, y_train)

# 未知のデータでの予測を行う
new = pd.DataFrame([[150, 700, 300, 0]], columns=x_train.columns)
model.predict(new)

model.score(x_test, y_test)

# MAEを求める
pred = model.predict(x_test)

# 平均絶対誤差の計算
mean_absolute_error(y_pred = pred, y_true = y_test)

# モデルの保存


# 散布図の作成 
# df2.plot(kind='scatter', x='SNS2', y='sales')
# plt.show()

# for name in df2.columns:
#     if name == 'cinema_id' or name == 'sales':
#         continue

#     df2.plot(kind='scatter', x=name, y='sales')
#     plt.show()
