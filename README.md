# A primar on Adversarial Examples
adversarial examples を用いた攻撃手法とその防御手法のいくつかを検証するためのプログラム.  
この検証実験のプログラムに関しては, 書籍の Appendix A にも補足説明を記載しているのでそちらも参照のこと.


## データ準備
CIFAR10 に関しては PyTorch の `torchvision.datasets.CIFAR10` クラスをそのまま使用しているので準備は必要ない.

GTSRB に関しては以下の手順で準備をする.
- [公式ダウンロードページ](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset#Downloads) のリンクを辿って `GTSRB_Final_Training_Images.zip` を `./data/` にダウンロード.
- zip ファイルを解凍.
- `./notebook/GTSRB_preprocessing.ipynb` で前処理を実施.
- `./data/GTSRB_processed/` が実験で使用するデータとなる.


## 環境構築
Docker が使える環境を想定している.  
自分が使う時はスクリプトを作成したり小さいモデルで試す場合は local で CPU 環境を用いて, 大きめのモデルを使う場合は Colaboratory の GPU 環境で notebook に移植したコードを実行している.
基本的に GPU enabled な PyTorch が使える環境があれば, notebook を

Docker image をビルドして環境を構築する.

```
docker build -t work -f Dockerfile .
```

作成した image を用いて container を作成する.

```
docker run -it --rm -v $PWD:/work -p 8888:8888 work
```

container に入ったら PyTorch や scikit-learn などがインストールされている `work` 環境に入る.

```
(in a docker container)
source activate work
```


## 検証実験
データに関係するクラスは `data.py`, モデルに関係するクラスは `model.py` で記述しているが, それ以外の主要なコードは全て `main.py` に集約されている.  
構築した Anaconda の `work` 環境内で, 各種オプションを付与してこの `main.py` を実行することで各種実験が実施される.

具体例として, cifar10 のデータ通常の学習をして fgsm で作成した adversarial examples を使ったテストをしたい場合は以下のように実行する.

```
(in a docker container)
python main.py --dataset cifar10 --is_train --model normal --train_method none --is_test --test_method fgsm
```

別の例として, GTSRB_processed データで学習したモデル `model_normal_ifgsm` で, rfgsm で作成した adversarial examples でカーネル密度推定の実験とランダムリサイズと 0-padding の実験をしたい場合は以下のように実行する.

```
(in a docker container)
python main.py --dataset GTSRB_processed --model_name_for_test model_normal_ifgsm --test_method fgsm --is_kde_test --is_random_crop_test
```


全オプションの説明は以下の通り.

- `--log_dir` (str)  
  `./log` と指定するとそのディレクトリにログファイルを出力する. 指定しなければ標準出力. デフォルトは指定なし.
- `--dataset` (enum)  
  `cifar10` or `GTSRB_processed`. デフォルトは `cifar10`.
- `--is_train` (bool)  
  セットするとモデルを学習する. デフォルトはセットなし.
- `--model` (enum)  
  `simple` or `normal` or `normalSAP`. デフォルトは `simple`.
- `--train_method` (enum)  
  `none` は clean データで学習でそれ以外だと adversarial training で学習. `none` or `fgsm` or `rfgsm` or `ifgsm` or `mifgsm`. デフォルトは `none`.
- `--epochs` (int)  
  学習時の epochs 数. デフォルトは `150`.
- `--batch_size` (int)  
  バッチサイズ. デフォルトは `32`.
- `--epsilon` (float)  
  摂動の大きさ. デフォルトは `4. /255`.
- `--alpha` (float)  
  R+FGSM のノイズの大きさ. デフォルトは `2. /255`.
- `--step` (int)  
  I+FGSM と MI+FGSM の繰り返し回数. デフォルトは `20`.
- `--use_atda_loss` (bool)  
  セットすると ATDA loss を使用する. デフォルトはセットなし.
- `--is_test` (bool)  
  セットするとテストを実行する. デフォルトはセットなし.
- `--model_name_for_test` (str)  
  テスト時に使用するモデル path を `model_none_simple` のように指定する. デフォルトは指定なし.
- `--test_method` (enum)  
  テストデータ作成時の攻撃手法. `none` or `fgsm` or `rfgsm` or `ifgsm` or `mifgsm`. デフォルトは `none`.
- `--is_kde_test` (bool)  
  セットするとカーネル密度推定の実験を実施. デフォルトはセットなし.
- `--is_random_crop_test` (bool)  
  セットするとランダムリサイズと 0-padding の実験を実施. デフォルトはセットなし.