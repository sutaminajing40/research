# 概要
- GAN(Generative Adversartial Network)を実装し、理解を深めるための実験
- 次のサイトを参考にさせていただきました。
  - [PytorchでGANを実装してみた](https://qiita.com/keiji_dl/items/45a5775a361151f9189d)

# 環境作成方法
- 以下のREADME.mdを参考にして作成してください。
[README.md](../../README.md)

# 起動方法
## モデルのトレーニング
- 以下のコマンドをルートディレクトリ(`/`)で実行してください。
```
python -m src.methods.GAN.train.train
```

## 訓練済みモデルから画像生成
### 単一モデルから画像生成
- 以下のコマンドをルートディレクトリ(`/`)で実行してください。
- [generate_image.py](generator/generate_image.py)の最終行にある関数の引数にモデル名を指定してください。
```
python -m src.methods.GAN.generator.generate_image
```

### 複数モデルから画像生成
- 以下のコマンドをルートディレクトリ(`/`)で実行してください。
- 複数モデルの中からまだ画像を生成していないものを自動で生成してくれるので、こちらのほうが便利です。
```
python -m src.methods.GAN.generator.generate_images
```

# 結果の表示方法
- 以下のコマンドをルートディレクトリ(`/`)で実行してください。
- その後ターミナルに表示されたリンクを開くと画像が表示され、比較ができます。
```
streamlit run src\methods\GAN\generator\display_images.py
```

# 結果

# 成果

# 参考文献
- [PytorchでGANを実装してみた](https://qiita.com/keiji_dl/items/45a5775a361151f9189d)