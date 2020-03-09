## What is this
データ分析のフローについて整理する用<br>
[nejumiさんのkaggle_memo](https://github.com/nejumi/kaggle_memo)<br>
[amatoneさんのkaggle_memo](https://github.com/amaotone/kaggle-memo)<br>


## Folder structure and About File
### Folder structure
```
.
├── README.md
└── code
    ├── 
    ├──
    ├── 
    └── 
```
### About File
- README.md
処理フローについて
- code
余裕があればpipeline作る。そのディレクトリだけは準備している

## 分析フロー
### 大前提
ライブラリとか
```
使用するライブラリ
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc

# 表示できるカラム数の設定を変更
pd.set_option('max_columns', 100)
# 表示できるレコード数の設定を変更
pd.set_option('max_rows', 100)
```

### EDA
#### まずは何よりデータを見てみる<br>
##### 読み込み
`df = read_csv('file_path', dtype={'カラム名': データ型, ...})`
##### 表示
`df.head()`<br>
`df.tail()`

##### 何を使って、何を予測するか


##### 必要であれば、連結・結合
- 連結：`df = pd.concat([df1, df2], axis=(0: 縦方向、1: 横方向)`
- 結合：`df = df1.merge(df2, on=['keyとなるカラム名'], how='(inner: 両方のテーブルに共通のもののみ, left: 左のテーブルにあるもののみ, right, outerなど)')`

#### データの概要を掴む
##### データの型
- 扱うデータ・各カラムはどんな型？
`df.info()` or `df.dtypes()`

##### 欠損あるか
- 欠損ではない要素の数の確認
`df.info()`
- 全体のレコードのうち有効数の割合を計算
    →欠損の割合もわかる
`df.count()/len(df)`

##### 要約統計量
- 平均分散最大最小四分位数などそこらへん
    →異常値などを削除するかどうかの判断材料にもなる
`df.describe()`
- 四分位数じゃ物足りないときは'percentiles'
`df.describe(percentiles=[0.1, 0.2, ..., 0.9])`

##### データの分布
- 正規分布？歪な分布？
`df['カラム名'].hist()`
- 対数変換かましてみる（要注意）
[Log-transform and its implications for data analysis](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4120293/)<br>
`np.log1p(df['カラム名']).hist()`

##### 数値変数
- GBDTだったらそのままでOK
大小関係のみ影響（数値間の幅はあまり関係がない）

- NNだったら標準化（対象の変数を0~1の範囲に変換）必須
```
from sklearn.preprocessing import MinMaxScale
scaler = MinMaxScaler()
scaler.fit(data)
scaler.transform(data)
scaler.fit_transform(data)
...
```

- 正規化（平均0、分散1に変換）

```
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(data)
scaler.transform(data)
scaler.fit_transform(data)
...
```

- その他
外れ値を無視するRobustScaler(`from sklearn.preprocessing import RobustScaler`)<br>
個々のデータポイントをそれぞれ異なるスケール変換するNormalizer(`from sklearn.preprocessing import Normalizer`)<br>

##### カテゴリ特徴の確認
文字列のままでは、処理を行なっていくことができないので、数値に変換する<br>
- ラベルエンコーディング
文字列を数値に変換<br>
```
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['encoded_カラム名'] = le.fit(df['カラム名'].values)

# 元に戻したいときは
df['decoded_カラム名'] = le.inverse_transform(df['encoded_カラム名'])
```

- OneHotエンコーディング
```
from sklearn.preprocessing import OneHotEncoder
# handle_unknown='ignore'で不明なカテゴリが生じたとき、無視される
oh = OneHotEncoder(handle_unknown='ignore')
oh.fit(df['カラム名'])
oh.transform(df['カラム名'])
oh.fit_transform(df['カラム名'])
...
```

- ターゲットエンコーディング
- ...


##### 時系列データの確認
- 時系列での傾向あるか


##### 相間あるかどうなのか
`corr = df['カラム名'].corr`
- ちなみに可視化はこんな感じ
`sns.heatmap(corr)`

### Feature Engineering


#### 欠損値
- 欠損値の削除
`df.dropna(how='(all: 全ての欠損の行または列が削除, any: どれか１つでも欠損があれば削除)', axis=(0: 行, 1: 列))`
- 欠損値を補間
`df.fillna(df.['カラム名'].mean())`: 平均値で補間<br>
`df.fillna(df.['カラム名'].median())`: 中央値で補間<br>
`df.fillna(df.['カラム名'].mode())`: 最頻値で補間<br>

#### 集約
- 'group'カラムの値ごとの'value'カラムの平均を計算する場合
`df.groupby('group')['value'].mean()` <br>
`df.groupby('group').['value'].agg(['mean])`<br>
上記2通り方法があるが、形式が異なるので、場面場面によって使い分けが必要

### Modeling
#### とりあえず突っ込む
#### validationはどうするか
##### Holdout
- 単純に分割
```
# XとyのデータをX_train, X_valid, y_train, y_testに分割する場合
# testデータのサイズを全体の1/10と指定

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=0)

```

##### Cross Validatoin
- KFold
```
from sklearn.model_selection import KFold
# 分割数を指定
kf = KFold(n_splits=2)

# 分割数を返す
# kf.get_n_splits(X)

for train_ind, valid_ind in kf.split(X):
     X_train, X_valid = X[train_ind], X[valid_ind]
     y_train, y_valid = y[train_ind], y[valid_ind]
     ...
```

- GroupKFold
説明変数にグループ要素があるとき（例えば、顧客・店舗など）。同じグループがtrainingとvalidationどちらにも存在してしまう場合、validationの結果が実際より高く見積もられることがある<br>
そういったことが怒らないように、あらかじめ、同じグループは同じデータセット（traininigかvalidationのどちらか）に分割するCVの切り方<br>
```
from sklearn.model_selection import GroupKFold

# 分割数を指定
gf = GroupKFold(n_splits=2)

# 分割'group'
for train_ind, valid_ind in gf.split(X, y, 'group'):
    X_train, X_valid = X[train_ind], X[valid_ind]
    y_train, y_valid = y[train_ind], y[valid_ind]
    ...
```

- StratifiedKFold
（主に分類タスクで）目的変数の分布に偏りがあるときに、分割した際にtraininigデータ、validationデータそれぞれにおいて、目的変数の分布が同じになるように分割するCVの切り方<br>
```
from sklearn.model_selection import StratifiedKFold
# 分割数を指定
sf = StratifiedKFold(n_splits=2)
for train_ind, valid_ind in sf.split(X, y):
    X_train, X_valid = X[train_ind], X[valid_ind]
    y_train, y_valid = y[train_ind], y[valid_ind]
    ...
```

- LeaveOneOut
```
from sklearn.model_selection import LeaveOneOut
loo = LearveOneOut()

# 分割数を返す
loo.get_n_splits(X)

# インデックスでtrainingとvalidationデータを参照する
for train_ind, valid_ind in loo.split(X):
    X_train, X_valid = X[train_ind], X[valid_ind]
    y_train, y_valid = y[train_ind], y[valid_ind]
    ...
```

#### cvの確認

#### 決定木系は重要度

#### 


