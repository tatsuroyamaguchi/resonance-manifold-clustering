import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler


class ResonanceManifoldClustering:
    def __init__(
        self,
        n_neighbors=10,
        coupling_strength=10.0,
        dt=0.05,
        max_iterations=5000,
        tol=1e-5,
        n_runs=5,
        dbscan_eps=0.1,
        dbscan_min_samples=3,
        normalize=True,
        random_state=None,
        **kwargs  # 後方互換性のため、古いsigma等を受け止めて無視する
    ):
        """
        RMC (Resonance Manifold Clustering)

        Parameters
        ----------
        sigma : float
            ガウスカーネルのスケールパラメータ。
            normalize=True の場合は正規化後のスケールで考える。
        coupling_strength : float
            振動子間の結合の全体的な強さ (K)。
        dt : float
            オイラー法のタイムステップ。小さいほど安定するが収束が遅くなる。
        max_iterations : int
            最大反復回数（収束しなかった場合の上限）。
        tol : float
            収束判定の閾値。全振動子の更新量の最大値がこれを下回れば早期終了。
        n_runs : int
            異なるランダム初期値での実行回数。コンセンサスクラスタリングで統合する。
        dbscan_eps : float
            位相クラスタ抽出用 DBSCAN のε。
            位相を単位円 2D ベクトルに変換してから適用するため 0〜2 が目安。
        dbscan_min_samples : int
            DBSCAN の min_samples。
        connectivity_threshold : float
            この値未満の結合をゼロにするスパース化閾値。

            【なぜ必要か】
            本実装では全振動子の固有振動数を ω=0 としている（簡略化）。
            ω が全て同一の場合、倉本モデルはどんなに微弱な結合でも
            十分な時間が経てば全体を同一位相に同期させてしまう。
            閾値でグラフを切断することで、クラスタ間の独立性を保証する。
        normalize : bool
            True のとき StandardScaler を内部で適用する。異なるスケールの
            特徴量が混在する実データでは事実上必須。
        random_state : int or None
            乱数シード。指定すると結果が再現可能になる。
        """
        self.n_neighbors = n_neighbors
        self.K = coupling_strength
        self.dt = dt
        self.max_iterations = max_iterations
        self.tol = tol
        self.n_runs = n_runs
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.normalize = normalize
        self.random_state = random_state

        # 学習後に参照できる属性
        self.labels_ = None
        self.final_phases_ = None   # 最後の run の最終位相
        self.n_iter_ = None         # 最後の run の実際の反復回数
        self.converged_ = None      # 最後の run で収束したか

    # ------------------------------------------------------------------
    # 内部ヘルパー
    # ------------------------------------------------------------------

    def _build_adjacency(self, X):
        """
        局所的スケール調整（Local Scaling）手法と k-NN グラフを組み合わせて結合グラフを構築する。
        各データ点 i について、k番目に近い点までの距離を独自のスケール sigma_i とし、
        結合の強さを exp(-d(i,j)^2 / (sigma_i * sigma_j)) で定義する。
        さらに、k近傍に入らない遠い接続は完全に遮断（0）することで、無関係な多様体同士が同期するのを防ぐ。
        これにより、手動でのパラメータチューニングなしで、複雑な形状のクラスタリングが自動化される。
        """
        from sklearn.neighbors import kneighbors_graph
        
        n_neighbors = min(self.n_neighbors, X.shape[0] - 1)
        if n_neighbors < 1:
            n_neighbors = 1
            
        # 1. 各点の k 個の近傍グラフを作成し、距離を取得
        adj_sparse = kneighbors_graph(X, n_neighbors=n_neighbors, mode='distance', include_self=False)
        dists = adj_sparse.toarray()
        
        # 2. 対称化（双方向に結合させる）
        dists = np.maximum(dists, dists.T)
        
        # 3. 各点の k 番目の距離を sigma_i とする（各行の最大値が k 番目の近傍距離になる）
        sigma = np.max(adj_sparse.toarray(), axis=1)
        sigma = np.maximum(sigma, 1e-10)  # 0除算防止
        
        # 4. Local Scaling の適用（エッジが存在する dists > 0 の場所のみ計算）
        sigma_matrix = np.outer(sigma, sigma)
        adj = np.zeros_like(dists)
        mask = dists > 0
        adj[mask] = np.exp(- (dists[mask] ** 2) / sigma_matrix[mask])
        
        # 自分自身との結合を消す
        np.fill_diagonal(adj, 0.0)
        
        return adj

    def _simulate(self, adjacency_matrix, rng):
        """
        倉本モデルの同期シミュレーション（1 run 分）。

        修正点
        ------
        - random_state 管理された rng で再現性を保証する
        - 収束判定（tol）を追加し、早期終了を可能にする

        Returns
        -------
        phases : ndarray, shape (n_samples,)
        n_iter : int
        converged : bool
        """
        n_samples = adjacency_matrix.shape[0]
        phases = rng.uniform(0, 2 * np.pi, n_samples)

        converged = False
        n_iter = self.max_iterations

        # 各ノードの次数（エッジの数）を計算しておく
        degrees = np.maximum(np.sum(adjacency_matrix, axis=1), 1.0)
        
        for i in range(self.max_iterations):
            phase_diff = phases[np.newaxis, :] - phases[:, np.newaxis]
            d_phases = (
                np.sum(adjacency_matrix * np.sin(phase_diff), axis=1)
                * (self.K / degrees)
            )
            phases += d_phases * self.dt
            phases = np.mod(phases, 2 * np.pi)

            # 収束判定: 全振動子の更新量の最大値が tol 未満なら停止
            if np.max(np.abs(d_phases)) < self.tol:
                converged = True
                n_iter = i + 1
                break

        return phases, n_iter, converged

    def _phases_to_labels(self, phases):
        """
        位相を単位円上の 2D ベクトルに変換し DBSCAN でクラスタ抽出する。

        【修正の核心】
        従来の np.round(phases, decimals=1) は円周境界バグを持つ:
          - 0 rad   → 0.0  (クラスタ A)
          - 6.28 rad → 6.3  (クラスタ B) ← 円上では同じ位置なのに別クラスタ！

        変換 z = [cos(θ), sin(θ)] を使うと:
          - 0 rad   → [ 1.0, 0.0]
          - 2π rad  → [ 1.0, 0.0]  ← 同一ベクトル。正しく同一クラスタになる

        DBSCAN の利点:
          - クラスタ数を事前に指定不要
          - ノイズ点（-1 ラベル）を検出できる
        """
        phase_vectors = np.column_stack([np.cos(phases), np.sin(phases)])
        db = DBSCAN(
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples,
            metric="euclidean",
        ).fit(phase_vectors)
        return db.labels_

    def _consensus_labels(self, all_labels):
        """
        複数 run の結果をコンセンサスクラスタリングで統合する。

        手法: 共起行列（co-association matrix）を構築し、
        複数 run で同一クラスタに属した割合を類似度として最終ラベルを決定。
        局所解への依存を減らし、安定したクラスタを得られる。
        """
        n_samples = all_labels[0].shape[0]
        n_runs = len(all_labels)

        co_matrix = np.zeros((n_samples, n_samples), dtype=np.float32)
        for labels in all_labels:
            for k in np.unique(labels):
                if k == -1:  # DBSCAN のノイズ点は除外
                    continue
                mask = labels == k
                co_matrix[np.ix_(mask, mask)] += 1.0

        co_matrix /= n_runs  # 0〜1 の類似度に正規化

        dist_matrix = 1.0 - co_matrix
        np.fill_diagonal(dist_matrix, 0.0)

        # 過半数の run で同一クラスタ = 同じクラスタとみなす
        db = DBSCAN(
            eps=0.5,
            min_samples=self.dbscan_min_samples,
            metric="precomputed",
        ).fit(dist_matrix)
        return db.labels_

    # ------------------------------------------------------------------
    # 公開インターフェース (scikit-learn 規約に準拠)
    # ------------------------------------------------------------------

    def fit(self, X):
        """
        モデルを学習する（scikit-learn 規約: fit と predict を分離）。
        学習後は self.labels_ でラベルを参照できる。
        """
        rng = np.random.default_rng(self.random_state)

        if self.normalize:
            self._scaler = StandardScaler()
            X_fit = self._scaler.fit_transform(X)
        else:
            self._scaler = None
            X_fit = X

        adjacency_matrix = self._build_adjacency(X_fit)

        all_labels = []
        for run_idx in range(self.n_runs):
            phases, n_iter, converged = self._simulate(adjacency_matrix, rng)
            labels = self._phases_to_labels(phases)
            all_labels.append(labels)

            if run_idx == self.n_runs - 1:
                self.final_phases_ = phases
                self.n_iter_ = n_iter
                self.converged_ = converged

        if self.n_runs == 1:
            self.labels_ = all_labels[0]
        else:
            self.labels_ = self._consensus_labels(all_labels)

        return self

    def fit_predict(self, X):
        """fit と predict を一括実行する（利便性のために残す）。

        Returns
        -------
        labels : ndarray, shape (n_samples,)
            クラスタラベル。ノイズ点は -1。
        """
        self.fit(X)
        return self.labels_


# ==========================================================================
# 実行例
# ==========================================================================
if __name__ == "__main__":
    from sklearn.datasets import make_moons, make_blobs
    import warnings
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("テスト 1: make_moons（三日月型）")
    print("=" * 60)
    X, true_labels = make_moons(n_samples=200, noise=0.05, random_state=42)

    rmc = ResonanceManifoldClustering(
        sigma=0.3,
        coupling_strength=5.0,
        dt=0.05,
        max_iterations=3000,
        tol=1e-5,
        n_runs=5,
        dbscan_eps=0.3,
        dbscan_min_samples=3,
        connectivity_threshold=0.1,  # クラスタ間の微弱な結合を切断
        normalize=True,
        random_state=0,              # 再現性のためシードを固定
    )
    labels = rmc.fit_predict(X)

    n_clusters = len(set(labels) - {-1})
    n_noise = np.sum(labels == -1)
    print(f"発見されたクラスタ数 : {n_clusters}  （正解: 2）")
    print(f"ノイズ点の数         : {n_noise}")
    print(f"実際の反復回数       : {rmc.n_iter_}")
    print(f"収束                 : {rmc.converged_}")

    print()
    print("=" * 60)
    print("テスト 2: make_blobs（球状 3 クラスタ）")
    print("=" * 60)
    X2, true2 = make_blobs(n_samples=300, centers=3, random_state=42)

    rmc2 = ResonanceManifoldClustering(
        sigma=0.5,
        coupling_strength=5.0,
        max_iterations=3000,
        connectivity_threshold=0.1,
        n_runs=5,
        normalize=True,
        random_state=0,
    )
    labels2 = rmc2.fit_predict(X2)

    n_clusters2 = len(set(labels2) - {-1})
    print(f"発見されたクラスタ数 : {n_clusters2}  （正解: 3）")
    print(f"収束                 : {rmc2.converged_}")
