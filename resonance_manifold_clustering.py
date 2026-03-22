import numpy as np
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler


class ResonanceManifoldClustering:
    def __init__(
        self,
        n_clusters=None,
        n_neighbors=10,
        coupling_strength=10.0,
        dt=0.05,
        max_iterations=5000,
        tol=1e-5,
        n_runs=5,
        dbscan_eps=0.1,
        dbscan_min_samples=3,
        connectivity_threshold=None,
        normalize=True,
        random_state=None,
    ):
        """
        RMC (Resonance Manifold Clustering)

        Parameters
        ----------
        n_clusters : int or None
            クラスタ数を事前に指定する場合に設定する。
            指定した場合、位相のクラスタ抽出に KMeans を、
            コンセンサス統合に AgglomerativeClustering を使用する。
            None（デフォルト）のときはクラスタ数を自動検出する DBSCAN を使用する。
            ノイズ点の概念が不要な場合や、クラスタ数が既知の場合に指定するとよい。
        n_neighbors : int
            Local Scaling に使う k-NN の k。
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
            位相クラスタ抽出用 DBSCAN のε（n_clusters=None のときのみ有効）。
            位相を単位円 2D ベクトルに変換してから適用するため 0〜2 が目安。
        dbscan_min_samples : int
            DBSCAN の min_samples（n_clusters=None のときのみ有効）。
        connectivity_threshold : float or None
            この値未満の結合重みをゼロにするスパース化閾値。None のとき無効。

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

        Notes
        -----
        旧バージョンの `sigma` パラメータは Local Scaling 方式へ移行したため廃止。
        `connectivity_threshold` は以前 **kwargs で黙って無視されていたが、
        現バージョンで正式なパラメータとして機能するよう修正した。
        """
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.K = coupling_strength
        self.dt = dt
        self.max_iterations = max_iterations
        self.tol = tol
        self.n_runs = n_runs
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.connectivity_threshold = connectivity_threshold
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

        n = X.shape[0]
        n_neighbors = min(self.n_neighbors, n - 1)

        # [FIX A] n_clusters 指定時は n_neighbors を適応的に制限する。
        # クラスタサイズ ≈ n / n_clusters のとき、k がクラスタサイズを超えると
        # 異なるクラスタの点を接続してしまい、倉本モデルが全体を同期させる。
        # k を「クラスタサイズの半分」以下に抑えることでクラスタ内接続を優先する。
        if self.n_clusters is not None and self.n_clusters > 1:
            intra_budget = max(2, n // self.n_clusters // 2)
            n_neighbors = min(n_neighbors, intra_budget)

        if n_neighbors < 1:
            n_neighbors = 1

        # 1. 各点の k 個の近傍グラフを作成し、距離を取得
        adj_sparse = kneighbors_graph(X, n_neighbors=n_neighbors, mode='distance', include_self=False)

        # [FIX 4] sigma は対称化前の有向グラフから計算する（各点の k 番目近傍距離）
        # .toarray() の再呼び出しを避けるため、最初に一度だけ変換して保持する
        dists_directed = adj_sparse.toarray()
        sigma = np.max(dists_directed, axis=1)
        sigma = np.maximum(sigma, 1e-10)  # 0除算防止

        # 2. 対称化（双方向に結合させる）
        dists = np.maximum(dists_directed, dists_directed.T)

        # 3. Local Scaling の適用（エッジが存在する dists > 0 の場所のみ計算）
        sigma_matrix = np.outer(sigma, sigma)
        adj = np.zeros_like(dists)
        mask = dists > 0
        adj[mask] = np.exp(-(dists[mask] ** 2) / sigma_matrix[mask])

        # 自分自身との結合を消す
        np.fill_diagonal(adj, 0.0)

        # [FIX 1] connectivity_threshold の決定:
        # ユーザー指定がある場合はそれを使用。
        # n_clusters 指定かつ未指定の場合は MST で自動計算（FIX B）。
        threshold = self.connectivity_threshold
        if threshold is None and self.n_clusters is not None and self.n_clusters > 1:
            # [FIX B] MST ベースの自動閾値:
            # グラフが正確に n_clusters 個の連結成分を持つように
            # 弱い接続を自動的に切断する。
            # これにより ω=0 の倉本モデルでも正しい分離が保証される。
            threshold = self._find_threshold_for_n_clusters(adj)

        if threshold is not None:
            adj[adj < threshold] = 0.0

        return adj

    def _spectral_labels(self, X_fit):
        """
        Zelnik-Manor & Perona (2004) の Self-Tuning Spectral Clustering を実装する。

        n_clusters が指定されている場合に呼ばれる高精度クラスタリング手法。

        【アルゴリズム概要】
        1. k-NN から各点の Local Scale σ_i を取得する
        2. 全点対の親和度を W_ij = exp(−d²/(σ_i·σ_j)) で計算する
        3. 正規化ラプラシアン L_sym = D^{-1/2} W D^{-1/2} の上位固有ベクトルを求める
        4. 固有ベクトル空間（行を正規化）で KMeans を実行する

        【倉本シミュレーションとの使い分け】
        倉本モデル（ω=0）はグラフの連結成分を同期させることで「分離」を担うが、
        多数クラスタではランダム初期位相の衝突（min_phase_gap < 0.1 rad）が
        KMeans を混乱させる。
        スペクトル埋め込みはこの問題を持たず、かつ非線形多様体以外でも
        高い精度を発揮するため、n_clusters 指定時の第一手法として採用する。

        Moons/Circles などの非線形多様体には引き続き倉本モデルを使用する
        （n_clusters=None のパス）。
        """
        from sklearn.neighbors import kneighbors_graph
        from sklearn.metrics.pairwise import euclidean_distances
        import scipy.linalg

        n = X_fit.shape[0]

        # sigma の計算: n_clusters に応じて k を適応的に決定
        # クラスタサイズ ≈ n / n_clusters が大きいときは k を大きく取ることで
        # クラスタ内の広い近傍情報を sigma に反映できる
        k_sigma = max(5, n // self.n_clusters)
        k_sigma = min(k_sigma, n - 1)
        adj_knn = kneighbors_graph(X_fit, n_neighbors=k_sigma, mode='distance', include_self=False)
        sigma = np.maximum(np.max(adj_knn.toarray(), axis=1), 1e-10)

        # 全点対の Local Scaling 親和度行列を構築
        dists = euclidean_distances(X_fit)
        sigma_mat = np.outer(sigma, sigma)
        W = np.exp(-(dists ** 2) / sigma_mat)
        np.fill_diagonal(W, 0.0)

        # 正規化ラプラシアン L_sym = D^{-1/2} W D^{-1/2}
        d = W.sum(axis=1)
        d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
        L_sym = d_inv_sqrt[:, None] * W * d_inv_sqrt[None, :]

        # 上位 n_clusters 固有ベクトルを取得（最大固有値から n_clusters 個）
        nc = min(self.n_clusters, n - 1)
        vals, vecs = scipy.linalg.eigh(L_sym, subset_by_index=[n - nc, n - 1])
        U = vecs[:, ::-1]  # 固有値の降順に並び替え

        # 行を L2 正規化（各点を単位超球面上に投影する）
        norms = np.linalg.norm(U, axis=1, keepdims=True)
        U = U / np.where(norms > 0, norms, 1.0)

        # 固有ベクトル空間で KMeans
        km = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10,
        )
        return km.fit_predict(U)

    def _find_threshold_for_n_clusters(self, adj):
        """
        MST（最小全域木）を利用して、グラフが正確に n_clusters 個の
        連結成分を持つような connectivity_threshold を自動計算する。

        【アルゴリズム】
        1. 類似度行列から距離行列 (−log(sim)) を構築する
        2. 現在の連結成分数を確認する（既に n_clusters 以上なら閾値不要）
        3. MST を求め、最大距離エッジを (追加カット数) 本除去する
        4. カット点の類似度を閾値として返す

        この手法により、ユーザーが connectivity_threshold を
        手動チューニングしなくても n_clusters 指定で正しい分離が得られる。
        """
        from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
        from scipy.sparse import csr_matrix

        n_cuts_total = self.n_clusters - 1
        if n_cuts_total <= 0:
            return None

        # 現在の連結成分数を確認（既に n_clusters 個あれば追加カット不要）
        n_current, _ = connected_components(csr_matrix(adj > 0), directed=False)
        additional_cuts = self.n_clusters - n_current
        if additional_cuts <= 0:
            return None  # 既に十分な成分数

        # −log(similarity) を距離として使用（similarity が高い ↔ 距離が短い）
        dist = np.where(adj > 0, -np.log(np.clip(adj, 1e-10, 1.0)), 0.0)
        mst = minimum_spanning_tree(csr_matrix(dist))
        mst_weights = np.sort(mst.data)[::-1]  # 大きい距離順（弱い接続順）

        if len(mst_weights) < additional_cuts:
            return 0.0

        # additional_cuts 番目に大きいエッジの距離 → 対応する類似度閾値
        cut_dist = mst_weights[additional_cuts - 1]
        sim_threshold = float(np.exp(-cut_dist)) + 1e-9

        return sim_threshold

    def _simulate(self, adjacency_matrix, rng):
        """
        倉本モデルの同期シミュレーション（1 run 分）。

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

        # 各ノードの次数（エッジの重み和）を計算しておく
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

    def _phases_to_labels(self, phases, rng):
        """
        位相を単位円上の 2D ベクトルに変換してクラスタ抽出する。

        【設計意図】
        単純な np.round(phases) は円周境界バグを持つ:
          - 0 rad   → 0.0  (クラスタ A)
          - 6.28 rad → 6.3  (クラスタ B) ← 円上では同じ位置なのに別クラスタ！

        変換 z = [cos(θ), sin(θ)] を使うと:
          - 0 rad   → [ 1.0, 0.0]
          - 2π rad  → [ 1.0, 0.0]  ← 同一ベクトル。正しく同一クラスタになる

        n_clusters が指定されている場合は KMeans を使用する。
        KMeans はノイズ点の概念を持たないため、全点がいずれかのクラスタに割り当てられる。
        None の場合は DBSCAN による自動検出（ノイズ点あり）。
        """
        phase_vectors = np.column_stack([np.cos(phases), np.sin(phases)])

        if self.n_clusters is not None:
            km = KMeans(
                n_clusters=self.n_clusters,
                random_state=int(rng.integers(0, 2**31)),
                n_init="auto",
            )
            return km.fit_predict(phase_vectors)
        else:
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
        複数 run で同一クラスタに属した割合を類似度として最終ラベルを決定する。
        局所解への依存を減らし、安定したクラスタが得られる。

        n_clusters が指定されている場合は AgglomerativeClustering を使用する。
        全点を n_clusters 個に確定割り当てするため、ノイズ点（-1）は生じない。
        None の場合は DBSCAN による自動検出（ノイズ点あり）。
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

        if self.n_clusters is not None:
            agg = AgglomerativeClustering(
                n_clusters=self.n_clusters,
                metric="precomputed",
                linkage="average",
            )
            return agg.fit_predict(dist_matrix)
        else:
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

        【手法の選択ロジック】
        ┌─────────────────────────────┬──────────────────────────────────────┐
        │ n_clusters 指定あり         │ Self-Tuning Spectral Clustering      │
        │                             │ （Local Scaling + 正規化ラプラシアン  │
        │                             │   固有ベクトル + KMeans）            │
        ├─────────────────────────────┼──────────────────────────────────────┤
        │ n_clusters=None（自動検出） │ 倉本モデル同期 + DBSCAN             │
        │                             │ （Moons・Circles 等の非線形多様体）  │
        └─────────────────────────────┴──────────────────────────────────────┘
        """
        rng = np.random.default_rng(self.random_state)

        if self.normalize:
            self._scaler = StandardScaler()
            X_fit = self._scaler.fit_transform(X)
        else:
            self._scaler = None
            X_fit = X

        # [FIX C+D] n_clusters 指定時は Self-Tuning Spectral Clustering を使用する。
        #
        # 旧実装（倉本 → KMeans → グラフ連結成分）の問題:
        #   1. n_clusters が多いとランダム初期位相が衝突し KMeans が誤分類する
        #   2. MST グラフ切断は単連結（single-linkage）と等価でチェーニングが生じる
        #
        # Zelnik-Manor & Perona (2004) の手法は:
        #   - σ_i を局所スケールとして自動決定するため手動調整不要
        #   - スペクトル空間の KMeans はチェーニングを起こさない
        #   - 非線形多様体にも対応しつつ球状クラスタにも強い
        if self.n_clusters is not None:
            self.labels_ = self._spectral_labels(X_fit)
            # final_phases_ を n_clusters=None パスと API 互換にするため 1 run だけ実施
            adjacency_matrix = self._build_adjacency(X_fit)
            phases, n_iter, converged = self._simulate(adjacency_matrix, rng)
            self.final_phases_ = phases
            self.n_iter_ = n_iter
            self.converged_ = converged
            return self

        # n_clusters=None: 倉本モデル + コンセンサス DBSCAN（Moons/Circles 向け）
        adjacency_matrix = self._build_adjacency(X_fit)
        all_labels = []
        for run_idx in range(self.n_runs):
            phases, n_iter, converged = self._simulate(adjacency_matrix, rng)
            labels = self._phases_to_labels(phases, rng)
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


# [FIX 3] demo.ipynb の `from rmc import RMC` に対応するエイリアスを定義
RMC = ResonanceManifoldClustering


# ==========================================================================
# 実行例
# ==========================================================================
if __name__ == "__main__":
    from sklearn.datasets import make_moons, make_blobs
    import warnings
    warnings.filterwarnings("ignore")

    # ------------------------------------------------------------------
    # テスト 1: n_clusters=None（自動検出）
    # ------------------------------------------------------------------
    print("=" * 60)
    print("テスト 1: make_moons（三日月型） — クラスタ数自動検出")
    print("=" * 60)
    X, _ = make_moons(n_samples=200, noise=0.05, random_state=42)

    rmc = ResonanceManifoldClustering(
        coupling_strength=5.0,
        dt=0.05,
        max_iterations=3000,
        tol=1e-5,
        n_runs=5,
        dbscan_eps=0.3,
        dbscan_min_samples=3,
        connectivity_threshold=0.1,
        normalize=True,
        random_state=0,
    )
    labels = rmc.fit_predict(X)

    n_found = len(set(labels) - {-1})
    n_noise = np.sum(labels == -1)
    print(f"発見されたクラスタ数 : {n_found}  （正解: 2）")
    print(f"ノイズ点の数         : {n_noise}")
    print(f"実際の反復回数       : {rmc.n_iter_}")
    print(f"収束                 : {rmc.converged_}")

    # ------------------------------------------------------------------
    # テスト 2: n_clusters=3（クラスタ数指定）
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("テスト 2: make_blobs（球状 3 クラスタ） — クラスタ数指定")
    print("=" * 60)
    X2, _ = make_blobs(n_samples=300, centers=3, random_state=42)

    rmc2 = ResonanceManifoldClustering(
        n_clusters=3,               # クラスタ数を明示指定
        coupling_strength=5.0,
        max_iterations=3000,
        connectivity_threshold=0.1,
        n_runs=5,
        normalize=True,
        random_state=0,
    )
    labels2 = rmc2.fit_predict(X2)

    n_found2 = len(set(labels2))    # n_clusters 指定時はノイズ点（-1）なし
    print(f"発見されたクラスタ数 : {n_found2}  （正解: 3）")
    print(f"ノイズ点の数         : {np.sum(labels2 == -1)}  （指定時は常に 0）")
    print(f"収束                 : {rmc2.converged_}")
