import pandas as pd
import numpy as np
from sklearn import tree
import math
import os

home_dir = os.path.expanduser("~")
test_path = os.path.join(home_dir, "デスクトップ", "機械学習", "signate", "SMBC Group GREENDATA Challenge 2024 Tutorial", "csv", "test.csv")
train_path = os.path.join(home_dir, "デスクトップ", "機械学習", "signate", "SMBC Group GREENDATA Challenge 2024 Tutorial", "csv", "train.csv")
test = pd.read_csv(test_path)
train = pd.read_csv(train_path)

# 欠損地の取得
def kesson_table(df):
    null_val = df.isnull().sum()
    percent = 100 * df.isnull().sum()/len(df)
    kesson_table = pd.concat([null_val, percent], axis=1)
    kesson_table_ren_columns = kesson_table.rename(columns = {0 : '欠損数', 1 : '%'})
    return kesson_table_ren_columns

print(kesson_table(train))
print(kesson_table(test))

# 欠損値の修復
# IndustryType列をstr型からint型に改良
# 1. train データに対してユニークなカテゴリを整数にマッピング
category_map = {category: idx for idx, category in enumerate(train['IndustryType'].unique())}

# 2. train と test の両方にそのマッピングを適用
train['IndustryType'] = train['IndustryType'].map(category_map)
test['IndustryType'] = test['IndustryType'].map(category_map)

# もし test に train にないカテゴリがある場合、'NaN' に変換されないようにする場合は、デフォルト値を設定
test['IndustryType'] = test['IndustryType'].fillna(-1)
# print(train['IndustryType'])

train['Latitude'] = train['Latitude'].fillna(train['Latitude'].median())
train['Longitude'] = train['Longitude'].fillna(train['Longitude'].median())
train['PrimaryNAICS'] = train['PrimaryNAICS'].fillna(train['PrimaryNAICS'].median())
test['Latitude'] = test['Latitude'].fillna(test['Latitude'].median())
test['Longitude'] = test['Longitude'].fillna(test['Longitude'].median())
test['PrimaryNAICS'] = test['PrimaryNAICS'].fillna(test['PrimaryNAICS'].median())

# 中央値
# 値が大きすぎるため自然対数に変換する
train['GHG_Direct_Emissions_10_in_metric_tons'] = train['GHG_Direct_Emissions_10_in_metric_tons'].fillna(train['GHG_Direct_Emissions_10_in_metric_tons'].median())
train['GHG_Direct_Emissions_11_in_metric_tons'] = train['GHG_Direct_Emissions_11_in_metric_tons'].fillna(train['GHG_Direct_Emissions_11_in_metric_tons'].median())
train['GHG_Direct_Emissions_12_in_metric_tons'] = train['GHG_Direct_Emissions_12_in_metric_tons'].fillna(train['GHG_Direct_Emissions_12_in_metric_tons'].median())
train['GHG_Direct_Emissions_13_in_metric_tons'] = train['GHG_Direct_Emissions_13_in_metric_tons'].fillna(train['GHG_Direct_Emissions_13_in_metric_tons'].median())
train['GHG_Direct_Emissions_10_in_metric_tons'] = np.log(train['GHG_Direct_Emissions_10_in_metric_tons'])
train['GHG_Direct_Emissions_11_in_metric_tons'] = np.log(train['GHG_Direct_Emissions_11_in_metric_tons'])
train['GHG_Direct_Emissions_12_in_metric_tons'] = np.log(train['GHG_Direct_Emissions_12_in_metric_tons'])
train['GHG_Direct_Emissions_13_in_metric_tons'] = np.log(train['GHG_Direct_Emissions_13_in_metric_tons'])
test['GHG_Direct_Emissions_10_in_metric_tons'] = test['GHG_Direct_Emissions_10_in_metric_tons'].fillna(test['GHG_Direct_Emissions_10_in_metric_tons'].median())
test['GHG_Direct_Emissions_11_in_metric_tons'] = test['GHG_Direct_Emissions_11_in_metric_tons'].fillna(test['GHG_Direct_Emissions_11_in_metric_tons'].median())
test['GHG_Direct_Emissions_12_in_metric_tons'] = test['GHG_Direct_Emissions_12_in_metric_tons'].fillna(test['GHG_Direct_Emissions_12_in_metric_tons'].median())
test['GHG_Direct_Emissions_13_in_metric_tons'] = test['GHG_Direct_Emissions_13_in_metric_tons'].fillna(test['GHG_Direct_Emissions_13_in_metric_tons'].median())
test['GHG_Direct_Emissions_10_in_metric_tons'] = np.log(test['GHG_Direct_Emissions_10_in_metric_tons'])
test['GHG_Direct_Emissions_11_in_metric_tons'] = np.log(test['GHG_Direct_Emissions_11_in_metric_tons'])
test['GHG_Direct_Emissions_12_in_metric_tons'] = np.log(test['GHG_Direct_Emissions_12_in_metric_tons'])
test['GHG_Direct_Emissions_13_in_metric_tons'] = np.log(test['GHG_Direct_Emissions_13_in_metric_tons'])
# print(train['GHG_Direct_Emissions_10_in_metric_tons'])


# 説明変数と予測変数の決定
features_one = train[['Latitude', 'Longitude', 'PrimaryNAICS', 'IndustryType', 'GHG_Direct_Emissions_10_in_metric_tons', 'GHG_Direct_Emissions_11_in_metric_tons', 'GHG_Direct_Emissions_12_in_metric_tons', 'GHG_Direct_Emissions_13_in_metric_tons']].values
target = train['GHG_Direct_Emissions_14_in_metric_tons'].values
target = np.log(target)
# 緯度、経度、NAICSコード、産業タイプ、2010、11、12、13年のCO2排気量

# 決定木の作成
my_tree_one = tree.DecisionTreeRegressor()
my_tree_one = my_tree_one.fit(features_one, target)
# 「test」の説明変数の値を取得
test_features = test[['Latitude', 'Longitude', 'PrimaryNAICS', 'IndustryType', 'GHG_Direct_Emissions_10_in_metric_tons', 'GHG_Direct_Emissions_11_in_metric_tons', 'GHG_Direct_Emissions_12_in_metric_tons', 'GHG_Direct_Emissions_13_in_metric_tons']].values
# 「test」の説明変数を使って「my_tree_one」のモデルを予測
my_prediction = my_tree_one.predict(test_features)
# NumPyを使用して指数関数を適用
my_prediction = np.exp(my_prediction)

# 予測データサイズを確認
# print(my_prediction.shape)
# print(my_prediction)

# idを取得
CityID = np.array(test["ID"]).astype(int)

# my_predictionとCityIDをデータフレームへ落とし込む
# CityID と my_prediction を行ごとに結合
data = list(zip(CityID, my_prediction))  # 行単位でデータをまとめる

# 列名なしの DataFrame を作成
my_solution = pd.DataFrame(data)

# my_tree_one.csvとして書き出し
my_solution.to_csv("my_tree_one.csv", index=False, header=False)



