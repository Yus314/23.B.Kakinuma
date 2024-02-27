# 23.B.Kakinuma

## 期間，タイトル，担当者
2022年4月-2024年3月，学習済みの潜在空間拡散モデルを用いた画像補完手法の一検討，柿沼祐介

## フォルダの詳細
* src: 実行ファイル
* Data/in: データセット


## 環境構築
仮想環境：venv
* OS：Windows10 22H2
* CPU：Intel Core i7-11700 2.50GHz
* GPU：NVIDIA GeForce RTX 3080 Ti
* Python 3.10.12
* PyTorch 2.0.0
* CUDA 11.3
* cuDNN 8.4.1

### pyenv のインストール方法
1. $git clone git://github.com/yyuu/pyenv.git ~/.pyenv
2. $ echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile
3. $ echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile
4. $echo 'eval "$(pyenv init -)"' >> ~/.bash_profile
5. $source ~/.bash_profile

### venv による環境構築法
```
pyenv install 3.10.12
pyenv local 3.10.12
python -m venv .venv
```

### poetry によるライブラリのインストール
pyproject.tomlがあるディレクトリで
```
poetry init
poetry install
```

`.devcontainer`にある`pyproject.toml`か'Dockerfile'をもとに仮想環境を構築してください．（不要なライブラリが含まれている可能性もあります）





## テスト
学習したモデルを用いてテストを行う場合は，次のコードを実行します．テスト時のオプションについては実行ファイルの中で指定し，実行時の引数は特に取りません．  
  
`python main.py`  
  
デフォルトでは以下の設定でテストを行います．
* データセット： BSD 200枚

テストコードを実行すると`Data/out`にディレクトリが作成されれば正常に動作しています．  
  
