import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
import graphviz

# CSVからデータ読み込み
a = pd.read_csv('Survived.csv')

# 特徴量ごとに生存者との関係をグラフにする

# 性別
sns.barplot(x = 'Sex', y = 'Survived', data = a)
plt.show()

# 年齢
a['AgeGroup'] = pd.cut(a['Age'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
sns.barplot(x = 'AgeGroup', y='Survived', data = a)
plt.show()

# Pclass
sns.barplot(x = 'Pclass', y = 'Survived', data = a)
plt.show()

# SibSp
sns.barplot(x = "SibSp", y = 'Survived', data = a)
plt.show()

# 特徴量xと正解データtに分割する
col = ['Sex','Age','Pclass','SibSp']

x = a[col]
x = pd.get_dummies(x, columns=['Sex'])
t = a['Survived']

# データ分割
x_train, x_test, y_train, y_test = train_test_split(x, t, test_size=0.2, random_state=0)

# 決定木モデル作成
model = tree.DecisionTreeClassifier(max_depth=3, random_state=0)
model.fit(x_train, y_train)

# 精度
print("Accuracy:", model.score(x_test, y_test))

# 決定木モデルの可視化
data = tree.export_graphviz(
    model,
    feature_names=x.columns,
    class_names=['Not Survived', 'Survived'],
    filled=True,
    rounded=True,
    special_characters=True
)

# 描画
# graph = graphviz.Source(data)
# graph.view()

# 特徴量の重要度を表示
importances = model.feature_importances_
print("特徴量の重要度:")
for feature, importance in zip(x.columns, importances):
    print(f"{feature}: {importance:.4f}")

