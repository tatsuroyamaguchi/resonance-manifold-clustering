# Resonance Manifold Clustering (RMC)

Resonance Manifold Clustering (RMC) は、非線形振動子の同期現象（倉本モデル）と多様体学習を組み合わせた新しいクラスタリング手法です。
K-Means 等の従来手法では分類が難しい「複雑な形状のデータ（三日月型やドーナツ型など）」に対しても、データ間の位相の同期を通じて自然なクラスタを抽出することができます。

## 概要 (Overview)

各データ点を個別の「振動子」とみなし、データ同士の距離が近いものほど強く結びつけることで全体の同期シミュレーションを行います。
シミュレーションの最終的な位相（Phase）を解析し、同じ位相に同期したデータのグループを 1 つのクラスタとして抽出します。

近傍スケールの自動調整（Local Scaling）と k-NN グラフを組み合わせているため、複雑な形状のクラスタに対してもパラメータのチューニングなしで自動適応します。

## インストール方法 (Installation)

### ローカルでのインストール（開発用）

リポジトリをクローンしたのち、フォルダ内で以下のコマンドを実行してください。

```bash
pip install -e .
```

### GitHub からの直接インストール

```bash
pip install git+https://github.com/あなたのユーザー名/リポジトリ名.git
```

## 使い方 (Usage)

インストール後は `from rmc import RMC` としてインポートできます。

### クラスタ数を自動検出する場合（デフォルト）

```python
from sklearn.datasets import make_moons
from rmc import RMC

X, _ = make_moons(n_samples=300, noise=0.08, random_state=42)

model = RMC(random_state=42)
labels = model.fit_predict(X)
# labels == -1 はノイズ点
```

### クラスタ数を事前に指定する場合

クラスタ数が既知の場合は `n_clusters` を指定してください。
ノイズ点（`-1`）の概念がなくなり、全点がいずれかのクラスタに割り当てられます。

```python
from sklearn.datasets import make_blobs
from rmc import RMC

X, _ = make_blobs(n_samples=300, centers=3, random_state=42)

model = RMC(n_clusters=3, random_state=42)
labels = model.fit_predict(X)
```

## パラメータ (Parameters)

| パラメータ | 型 | デフォルト | 説明 |
|---|---|---|---|
| `n_clusters` | int \| None | `None` | クラスタ数。`None` のとき DBSCAN で自動検出。指定時は KMeans / AgglomerativeClustering を使用 |
| `n_neighbors` | int | `10` | Local Scaling と k-NN グラフ構築に使用する近傍点の数 |
| `coupling_strength` | float | `10.0` | 振動子間の結合の全体的な強さ（K） |
| `dt` | float | `0.05` | シミュレーションのタイムステップ。小さいほど安定するが低速になる |
| `max_iterations` | int | `5000` | 同期シミュレーションの最大反復回数 |
| `tol` | float | `1e-5` | 収束判定の閾値。全振動子の更新量がこれを下回れば早期終了 |
| `n_runs` | int | `5` | 異なる初期値でのシミュレーション回数。コンセンサスクラスタリングで統合し安定性を高める |
| `dbscan_eps` | float | `0.1` | 位相クラスタ抽出用 DBSCAN の ε（`n_clusters=None` のときのみ有効） |
| `dbscan_min_samples` | int | `3` | DBSCAN の min_samples（`n_clusters=None` のときのみ有効） |
| `connectivity_threshold` | float \| None | `None` | この値未満の結合重みをゼロにするスパース化閾値。クラスタ間の独立性を保証するために有効 |
| `normalize` | bool | `True` | `True` のとき StandardScaler を内部で適用する。異なるスケールの特徴量が混在する実データでは事実上必須 |
| `random_state` | int \| None | `None` | 乱数シード。指定すると結果が再現可能になる |

### `connectivity_threshold` について

本実装では全振動子の固有振動数を ω=0 に固定しています（簡略化）。
ω が全て同一の場合、倉本モデルはどんなに微弱な結合でも十分な時間が経てば全体を同一位相に同期させてしまいます。
`connectivity_threshold` でグラフを切断することで、クラスタ間の独立性を明示的に保証できます。

## 学習後に参照できる属性 (Attributes after fitting)

| 属性 | 説明 |
|---|---|
| `labels_` | 各データ点のクラスタラベル。ノイズ点は `-1`（`n_clusters` 指定時は `-1` なし） |
| `final_phases_` | 最後の run における最終位相（ラジアン） |
| `n_iter_` | 最後の run での実際の反復回数 |
| `converged_` | 最後の run で収束判定を満たしたか |

## バージョン履歴 (Changelog)

### 現バージョン
- `n_clusters` パラメータを追加。クラスタ数の自動検出と事前指定を切り替え可能になった
- `connectivity_threshold` を正式なパラメータとして修正（以前は `**kwargs` で黙って無視されていた）
- `RMC` を `ResonanceManifoldClustering` のエイリアスとして追加
- `sigma` パラメータを廃止（Local Scaling 方式へ移行済み）
- 位相クラスタ抽出の円周境界バグを修正（`np.round` → 単位円ベクトル + DBSCAN）

## ライセンス (License)

MIT License
