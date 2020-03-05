## What is this
データ分析のフローについて書いてみる

## Folder structure and About File
```
たぶん階層的になる、なにか作る
.
├── README.md
├── 
│   ├── 
│   ├──
│   ├── 
│   └──
└── 
    ├──
    ├── 
    ├──
    └── 
```

## 分析フロー
### 大前提
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
`df.head()`
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
- 異常値と考えられるものを含むかどうか除去するかどうか
`df.describe()`
- 四分位数じゃ物足りない
`df.describe(percentiles=[0.1, 0.2, ..., 0.9])`

##### データの分布
- 正規分布？歪な分布？

- 対数変換かましてみる（要注意）
[Log-transform and its implications for data analysis](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4120293/)

##### 数値変数
- GBDTだったらそのままで
- NNだったら標準化

##### カテゴリ特徴の確認
- そのままでは使えない。変換してみる

##### 時系列データの確認
- 時系列での傾向あるか

##### 相間あるかどうなのか
df['カラム名'].corr

### Feature Engineering


#### 欠損値
#### 結合
#### 

### Modeling
#### とりあえず突っ込む
#### validationはどうするか
#### cvの確認
#### 決定木系は重要度
#### 


