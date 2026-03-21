# Resonance Manifold Clustering (RMC)

Resonance Manifold Clustering (RMC) は、非線形振動子の同期現象（倉本モデル）と多様体学習を組み合わせた新しいクラスタリング手法です。
K-Means等の従来の手法では分類が難しい「複雑な形状のデータ（三日月型やドーナツ型など）」に対しても、データ間の位相の同期を通じて自然なクラスタを抽出することができます。

## 概要 (Overview)
このモジュールは、各データ点を個別の「振動子」とみなし、データ同士の距離が近い（ガウスカーネルでの類似度が高い）ものほど強く結びつけることで全体の同期シミュレーションを行います。
シミュレーションの最終的な位相（Phase）を検証し、同じ位相に同期したデータのグループを1つのクラスタとして抽出します。

## インストール方法 (Installation)

### ローカルでのインストール（開発用）
このリポジトリをダウンロード・クローンしたのち、フォルダ内で以下のコマンドを実行すると、ご自身のPython環境にインストールされます。
```bash
pip install -e .
```

### GitHubからの直接インストール
リポジトリをGitHubにプッシュして公開した後は、以下のコマンドでどのPCからでも直接インストール可能です。
```bash
pip install git+https://github.com/あなたのユーザー名/リポジトリ名.git
```

## 使い方 (Usage)
インストール後は、どこからでも `from rmc import RMC` として使用できます。
以下は簡単なデモコード（scikit-learn のデータセットを使用した例）です。

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# RMCのインポート
from rmc import RMC

# 1. データの準備（三日月型のダミーデータ）
X, _ = make_moons(n_samples=300, noise=0.08, random_state=42)

# 2. モデルの初期化とパラメータ設定
model = RMC(
    sigma=0.3,                   # 類似度の広がりを決定するパラメータ
    coupling_strength=5.0,       # 振動子間の引力の強さ
    dt=0.05,                     # シミュレーションのタイムステップ
    max_iterations=3000,         # 最大反復回数
    n_runs=3,                    # 安定化のための反復シミュレーション回数
    connectivity_threshold=0.1,  # 弱い結合を切り離す閾値
    random_state=42
)

# 3. クラスタリングの実行
labels = model.fit_predict(X)

# 4. 可視化
# ノイズ（-1）を除外した有効なクラスタだけをプロット
valid = labels != -1
plt.scatter(X[valid, 0], X[valid, 1], c=labels[valid], cmap='tab10')
plt.show()
```

## パラメータについて (Parameters)
主なパラメータは以下の通りです。

- `sigma` (float): ガウスカーネルの分散値。各データ点が「どれくらい遠くまで影響を及ぼすか」を調整します。
- `coupling_strength` (float): すべての振動子どうしの結びつきの全体的な強さを決定します。
- `dt` (float): シミュレーションループ1回あたりの時間の進み具合。小さくすると安定しますが、処理に時間がかかります。
- `max_iterations` (int): 同期シミュレーションの最大反復回数です。
- `connectivity_threshold` (float): この値以下の弱い類似度（結合）はスパース化して0にします。これにより、無関係なクラスタ同士が同期してしまう長距離同期を防ぐ効果があります。
- `n_runs` (int): 異なる初期値で複数回シミュレーションを行い、その結果をコンセンサス（多数決）で統合することで安定性を高めます。
- `dbscan_eps` / `dbscan_min_samples`: 同期した後の最終的な位相データから、さらにクラスタを抽出する際に使用する DBSCAN のパラメータです。

## ライセンス (License)
MIT License (or your specified license)
