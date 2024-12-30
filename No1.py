import pandas as pd
import numpy as np
import os

home_dir = os.path.expanduser("~")
test_path = os.path.join(home_dir, "デスクトップ", "機械学習", "signate", "SMBC Group GREENDATA Challenge 2024 Tutorial", "csv", "test.csv")
train_path = os.path.join(home_dir, "デスクトップ", "機械学習", "signate", "SMBC Group GREENDATA Challenge 2024 Tutorial", "csv", "train.csv")
test = pd.read_csv(test_path)
train = pd.read_csv(train_path)
#index_colは1行目はないものとしてみてくれるもの。
# print(train.shape) #(4655, 21)
# print(test.shape) #508, 20)
# print(train.describe())
# print(test.describe())

# 欠損地の取得
def kesson_table(df):
    null_val = df.isnull().sum()
    percent = 100 * df.isnull().sum()/len(df)
    kesson_table = pd.concat([null_val, percent], axis=1)
    kesson_table_ren_columns = kesson_table.rename(columns = {0 : '欠損数', 1 : '%'})
    return kesson_table_ren_columns

print(kesson_table(train))
print(kesson_table(test))

train['Latitude'] = train['Latitude'].fillna(train['Latitude'].median())
train['Longitude'] = train['Longitude'].fillna(train['Longitude'].median())
train['FIPScode'] = train['FIPScode'].fillna(train['FIPScode'].median())
train['PrimaryNAICS'] = train['PrimaryNAICS'].fillna(train['PrimaryNAICS'].median())
test['Latitude'] = test['Latitude'].fillna(test['Latitude'].median())
test['Longitude'] = test['Longitude'].fillna(test['Longitude'].median())
test['FIPScode'] = test['FIPScode'].fillna(test['FIPScode'].median())
test['PrimaryNAICS'] = test['PrimaryNAICS'].fillna(test['PrimaryNAICS'].median())

from sklearn import tree

# trainの目的変数と説明変数の値を取得
target = train['GHG_Direct_Emissions_14_in_metric_tons'].values
features_one = train[['Latitude', 'Longitude', 'FIPScode', 'PrimaryNAICS']].values
# 緯度,経度,群のFIPSコード,主要NAICSコード
# 決定木の作成
my_tree_one = tree.DecisionTreeRegressor()
my_tree_one = my_tree_one.fit(features_one, target)
# 「test」の説明変数の値を取得
test_features = test[['Latitude', 'Longitude', 'FIPScode', 'PrimaryNAICS']]
# 「test」の説明変数を使って「my_tree_one」のモデルを予測
my_prediction = my_tree_one.predict(test_features)

# 予測データサイズを確認
print(my_prediction.shape)
print(my_prediction)

# idを取得
CityID = np.array(test["ID"]).astype(int)

# my_predictionとCityIDをデータフレームへ落とし込む
# CityID と my_prediction を行ごとに結合
data = list(zip(CityID, my_prediction))  # 行単位でデータをまとめる

# 列名なしの DataFrame を作成
my_solution = pd.DataFrame(data)

# my_tree_one.csvとして書き出し
my_solution.to_csv("my_tree_one.csv", index=False, header=False)

