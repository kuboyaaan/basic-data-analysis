## What is this
データ分析のフローについて整理する用<br>
[nejumiさんのkaggle_memo](https://github.com/nejumi/kaggle_memo)<br>
[amatoneさんのkaggle_memo](https://github.com/amaotone/kaggle-memo)<br>
[【随時更新】Kaggleテーブルデータコンペできっと役立つTipsまとめ](https://naotaka1128.hatenadiary.jp/entry/kaggle-compe-tips)<br>
[Kaggleに登録したら次にやること ～ これだけやれば十分闘える！Titanicの先へ行く入門 10 Kernel ～](https://qiita.com/upura/items/3c10ff6fed4e7c3d70f0)<br>

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
- README.md<br>
処理フローについて<br>
- code<br>
余裕があればpipeline作る。そのディレクトリだけは準備している<br>

## 分析フロー
### Task
- 回帰: 各データにおいて、対応する目的変数を連続値で予測するタスク<br>
（例)土地や建物の情報が与えられ、その不動産の価値を予測するタスク<br>

- 分類: 各データがどのクラスに属するかを予測するタスク<br>
（例1）メールがスパムであるかどうかの2値分類タスク<br>
（例2）犬の画像が与えられ、その犬種を予測するタスク（柴犬、パグ、プードル、、、）<br>

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
`df = read_csv('file_path', dtype={'カラム名': データ型, ...})`<br>
file_path: ここに読み込むcsvファイルまでのpathを記述する（tsv, .csv.zipなど様々な形式のファイルを読み込むことができる）<br>
dtype: 各カラムの値のデータ型を指定できる<br>

##### 表示
`df.head()`<br>
`df.tail()`
.head: デフォルトでレコード上位5つを表示させることができる<br>
.tail: デフォルトでレコード下位5つを表示させることができる<br>
↑()内に数字を指定することで表示する数を変更できる<br>

##### 何を使って、何を予測するか
データを観察したり、以下から説明するメソッドを使用して、データの特徴を掴む<br>
1. どういったデータを使用して、何を予測するのか<br>
2. それぞれのカラムはどういった傾向があるのかなど<br>
数値： 最大値最小値分布など<br>
カテゴリ：最頻値unique数など<br>
日付：トレーニング/テストデータそれぞれの期間について、周期性・日付固有のイベントの影響を受けるか（他の特徴量と合わせて確認してみる）など<br>

##### 必要であれば、連結・結合
複数のファイルで渡されたものや、処理の過程で分割したテーブルを連結・結合する際に使用<br>
- 連結：`df = pd.concat([df1, df2], axis=0)`<br>
df1, df2: 連結するデータフレーム形式のデータ<br>
axis: 0→縦方向、1→横方向<br>
- 結合：`df = df1.merge(df2, on=['keyとなるカラム名'], how='(inner: 両方のテーブルに共通のもののみ, left: 左のテーブルにあるもののみ, right, outerなど)')`<br>
on: ここに指定したカラムの値を用いてdf1とdf２が結合される。単数でも複数でも可<br>
同じ意味のkeyであるがカラム名が異なる場合は、`left_on='df1のカラム名', right_on='df2のカラム名'`で指定する<br>
how: 結合方法。inner→両方のテーブルに共通するもののみを残す。left→左のテーブル（上記だとdf1がそれに当たる）に存在するテーブルのみ残す。right→その逆。outer→両方に存在するテーブルを全て残す。<br>

#### データの概要を掴む
##### データの型
- 扱うデータ・各カラムはどんな型？<br>
`df.info()` or `df.dtypes`<br>
df.info(): データサイズやメモリ、各カラムのデータ型などを確認<br>
df.dtypes(): 各カラムの型を確認。`df['カラム名'].dtype`で指定カラムのみの型を確認することもできる<br>

##### 欠損あるか
- 欠損ではない要素の数の確認<br>
`df.info()`<br>
- 全体のレコードのうち有効数の割合を計算<br>
    →欠損の割合もわかる<br>
`df.count()/len(df)`<br>
df.count(): 有効数（欠損ではない値の数）を確認できる。<br>
len(df): len()でサイズを計算できるので、df全体のレコード数を計算<br>

##### 要約統計量
- 平均分散最大最小四分位数などそこらへん<br>
    →異常値などを削除するかどうかの判断材料にもなる、かも<br>
`df.describe()`<br>
- ↑四分位数じゃ物足りないときは'percentiles'<br>
`df.describe(percentiles=[0.1, 0.2, ..., 0.9])`
- 異常値の削除はclipping
```
upper, lower = np.percentile(df['カラム名'], [1, 99])
y = np.clip(df['カラム名'], upper, lower)
```

##### データの分布
- 正規分布？歪な分布？<br>
スパイク見つけるためにヒストグラム書こう (これは絶対)from[ML_Bearさん](https://naotaka1128.hatenadiary.jp/entry/kaggle-compe-tips)
`df['カラム名'].hist()`<br>
- 対数変換（要注意）<br>
[Log-transform and its implications for data analysis](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4120293/)<br>
`np.log1p(df['カラム名']).hist()`<br>

##### 数値変数
- GBDTだったらそのままでOK<br>
大小関係のみ影響（数値間の幅はあまり関係がない）<br>

- NNだったら標準化（対象の変数を0~1の範囲に変換）必須<br>
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
- Labelエンコーディング<br>
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

- Targetエンコーディング<br>
- Frequencyエンコーディング<br>
...<br>


##### 時系列データの確認
- datetime64型に変換<br>
```
# datetimeに時系列情報が格納されているとする
df['datetime'] = pd.to_datetime(df['datetime'])
df['month'] = df['datetime'].dt.month
...
```

- 時系列での傾向あるか
```
# 可視化例
# datetimeカラムに時系列情報（時間など）があるとする

import matplotlib.pyplot as plt
import matplotlib

fig, ax = plt.subplots(figsize=(15, 8))
dates = matplotlib.dates.date2num(df.loc[:, 'datetime'])
ax.plot_date(dates, df.loc[:, 'feature1'], '-', color='tab: brown', label='feature1', alpha=0.5)
ax2.plot_date(dates, df.loc[:, 'feature2'], '-', color='tab: cyan', label='feature2', alpha=0.5)
ax.legend(['feature1', 'feature2']);
```
- 時系列で特徴量をづらす
idごとにグループ化して、'target'カラムの値を'size'分づらす
```
size=7
df.groupby(['id'])['target'].transform(lambda x: x.shift(size))
```

- 指定期間での平均・分散...を計算
'size'期間での平均などの各種統計量を計算できる
`df.groupby(['id'])['target'].transform(lambda x: x.rolling(size)['mean'])`
'mean'の他に'std', 'max', 'min'などを計算できる


- 月や時間など周期的なもの
[Qiita](https://qiita.com/shimopino/items/4ef78aa589e43f315113)を参考にしています
```
# 'month'カラムに1~12の値が格納されている場合
def encode(df, col):
    df[col + '_cos'] = np.cos(2 * np.pi * df[col]/df[col].max())
    df[col + '_sin'] = np.sin(2 * np.pi * df[col]/df[col].max())
    return df
df = encode(df, 'month')
```


##### 相間あるかどうなのか
`corr = df['カラム名'].corr`<br>
- ちなみに可視化はこんな感じ<br>
`sns.heatmap(corr)`<br>

##### 散布図行列
`pd.scatter_matrix(df)`で一発で散布図行列を可視化することができる

### Feature Engineering

#### 欠損値
- 欠損値の削除<br>
`df.dropna(how='(all: 全ての欠損の行または列が削除, any: どれか１つでも欠損があれば削除)', axis=(0: 行, 1: 列))`<br>
- 欠損値を補間<br>
`df.fillna(df.['カラム名'].mean())`: 平均値で補間<br>
`df.fillna(df.['カラム名'].median())`: 中央値で補間<br>
`df.fillna(df.['カラム名'].mode())`: 最頻値で補間<br>

#### データクリーニング
- 特徴量ゼロのカラムを消す from[ML_Bearさん](https://naotaka1128.hatenadiary.jp/entry/kaggle-compe-tips)<br>
1種類の値しかはいっていないカラムなど<br>
```
remove = []
for col in df.columns:
    if df[col].std()==0:
        remove.append(col)
df.drop(remove, axis=1, inplace=True)
```
- （主に）時系列データにおいて、長期間0となっているデータはノイズになる可能性<br>
需要予測では、まだ販売されていない期間など<br>
[ASHRAEコンペ]()でもじゅうようだったぽい<br>
```
# コード例（もっと良い実装方法は存在するはず）

df['shift_1'] = df.groupby(['id'])['target'].transform(lambda x: x.shift(1))
df['shift_2'] = df.groupby(['id'])['target'].transform(lambda x: x.shift(2))
...
df['shift_n'] = df.groupby(['id'])['target'].transform(lambda x: x.shift(n))

# 全部0　ではないものを取り出す
# '~'は否定を表す
df = df[~((df.shift_1==0) & (df.shift_2==0) ...& (df.shift_n==0))]
```

#### 集約
- 'group'カラムの値ごとの'value'カラムの平均を計算する場合<br>
`df.groupby('group')['value'].mean()` <br>
`df.groupby('group').['value'].agg(['mean])`<br>
上記2通り方法があるが、形式が異なるので、場面場面によって使い分けが必要<br>
'mean'の他にも'sum', 'count', 'max', 'min',...<br>

### その他の特徴量操作
- 関係のありそうなカテゴリ特徴を明示的関連づける
```
def make_relation(df):
    df['relation_feature'] = list(map(lambda x, y: str(x) + '_' + str(y), df['feature1'], df['feature2']))

    all_feature = list(df['relation_feature'].unique())
    feature_map = dict(zip(all_feature, np.arange(len(all_feature))))

    df['relation_category'] = df['relation_feature'].map(feature_map)
    return df

df = make_relation(df)
```

### Modeling
- LightGBM
```
import lightgbm as lgb

dtrain ＝lgb.Dataset(train_df['features'], train_df['target'])
dvalid = lgb.Dataset(valid_df['features'], valid_df['target'])
dtest = lgb.Dataset(test_df['features'])

model = lgb.train(params, dtrain, valid_sets=(dtrain, dvalid), num_boost_round=, early_stopping_rounds=, verbose_eval=, ...)
pred = model.predict(dtest, ...)
```
- XGBoost
```
import xgboost as xgb

dtrain = xgb.DMatrix(train_df['features'], train_df['target'])
dvalid = xgb.DMatrix(valid_df['features'], valid_df['target'])
dtest = xgb.DMatrix(test_df['features'])

watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
model = xgb.train(params, dtrain, num_round, evals=watchlist)
pred = model.predict(dtest)
```
パラメータについては[こちらのQiita](https://qiita.com/FJyusk56/items/0649f4362587261bd57a)がわかりやすい

その他
- CatBoost
- NN
など


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
- lightGBM
```
lgb.plot_importance(model, ...)
```

- xgbBoost
```
xgb.plot_importance(model)
```
#### 


