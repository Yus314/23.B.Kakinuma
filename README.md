# 23.B.Kakinuma

## 期間，タイトル，担当者
2022年4月-2024年3月，学習済みの潜在空間拡散モデルを用いた画像補完手法の一検討，柿沼祐介

## フォルダの詳細
* src: 実行ファイル
* Data/in: データセット


## 環境構築
仮想環境：venv
* OS：Ubuntu22.04
* CPU：Intel Core i7-11700 2.50GHz
* GPU：NVIDIA GeForce RTX 3080 Ti
* Python 3.10.12
* PyTorch 2.0.0
* CUDA 11.3
* cuDNN 8.4.1



### poetry によるライブラリのインストール
poetry をインストールし、
pyproject.tomlがあるディレクトリで
```
poetry shell
```
作成された仮想環境を実行
[参考](https://zenn.dev/claustra01/articles/0d8efd08905526)
```
poetry init
poetry install
```

`.devcontainer`にある`pyproject.toml`か'Dockerfile'をもとに仮想環境を構築してください．（不要なライブラリが含まれている可能性もあります）





## テスト
学習したモデルを用いてテストを行う場合は，次のコードを実行します．テスト時のオプションについては実行ファイルの中で指定し，実行時の引数は特に取りません．  
  
`python main.py`  

## エラーが出た時の資料
[libcuda.so: cannot open shared object file](https://qiita.com/cacaoMath/items/811146342946cdde5b83)
[poetry](https://zenn.dev/canonrock/articles/poetry_basics)
  
デフォルトでは以下の設定でテストを行います．
* データセット： BSD 200枚

テストコードを実行すると`Data/out`にディレクトリが作成されれば正常に動作しています．  
  
