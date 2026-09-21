# Taremin Cloth 関連研究・先進アルゴリズム調査書 (Related Research & Advanced Techniques)

本ドキュメントは、Taremin Cloth の物理コア、自己衝突判定、拘束収束、およびGPU最適化に関連する学術論文・最先端技術（SOTA）を体系的にまとめた調査書です。  
会話初期のサーベイで挙げられた重要技術を含め、理論背景、メリット・デメリット、および適用可能性を網羅して記録します。

---

## 1. 幾何学的枝切り・自己衝突カリング (Self-Collision Culling)

### 1.1 離散曲率解析による事前カリング (Discrete Curvature Culling / Laplace-Beltrami)
* **代表文献**:
  * **Provot 1997**: *"Collision and self-collision handling in cloth model dedicated to design garments"* (Graphics Interface 1997 / Eurographics Workshop on Computer Animation and Simulation 1997)
  * **Volino & Magnenat-Thalmann 1994**: *"Efficient Self-collision Detection on Smoothly Discretized Surface Animations using Geometrical Shape Properties"* (Computer Graphics Forum, Vol. 13, No. 3, pp. 155–166)
    * [DOI: 10.1111/1467-8659.1330155](https://doi.org/10.1111/1467-8659.1330155)
  * **Tang, Curtis, Yoon, Manocha 2008 / 2009**: *"ICCD: Interactive Continuous Collision Detection between Deformable Models using Connectivity-Based Culling"* (ACM SPM 2008 / IEEE TVCG 2009, Vol. 15, No. 4)
  * **近年の動向 (2024–2026)**: 離散外微分（Discrete Differential Geometry）に基づく **Laplace-Beltrami 平均曲率法線（$`\Delta \mathbf{x}_i \approx H \mathbf{n}`$）** を用いて、衣服の平坦領域をリアルタイムに事前判定する手法。
* **幾何学的理論**:
  * メッシュ上の連結領域 $S$ において、領域内の面法線ベクトルが単位球上で半頂角 $`\theta < \pi/2`$（90度未満）の法線コーン（Normal Cone）内に収まり、かつ領域の境界輪郭が正射影で自己交差しない場合、**幾何学的に領域 $S$ の内部で局所的な自己交差・自己衝突は発生し得ない**という定理。
  * 衣服メッシュにおいて、シワの挟み込みが物理的に起こり得るのは**曲率が極めて高い（法線が急激に反転している）領域のみ**です。
* **Taremin Cloth への適用設計案**:
  1. **Laplace-Beltrami / 局所二面角による平坦度判定**:
     - 法線計算パス（`compute_normals.wgsl`）において、自頂点とその1ホップ隣接頂点の離散ラプラシアン $`\|\Delta \mathbf{x}_i\| = \|\sum_j w_{ij} (\mathbf{x}_j - \mathbf{x}_i)\|`$ を評価（余接重みまたは一様重み）。
     - 曲率が閾値未満（ほぼ平坦）である頂点スレッドは、`self_collision.wgsl` の能動的なセル探索ループを先頭でスキップ（Early Exit）。
  2. **期待される効果**:
     - 高密度メッシュにおいて、シワの寄っていない平坦領域（メッシュ全体の 60%〜80%）の探索・Möller–Trumbore交差判定が完全にバイパスされ、**自己衝突の計算量を 7〜8 割削減**できる可能性があります。
  3. **Uniform Grid アーキテクチャにおける構造的ミスマッチと不採用の理由（重要）**:
     - **すり抜け貫通（精度低下）の不可避なリスク**: スカートの「前生地（平坦）」と「後ろ生地（平坦）」が接触する場合など、同一パーツ内でも局所曲率が低いまま接近するケースにおいて、曲率だけで探索をスキップすると**生地同士がノーガードですり抜けて裏表が反転する大破綻**を招く。
     - **高速化の相殺ジレンマ**: 上記のすり抜けを防ぐためには「平坦であっても空間セルを走査してトポロジー距離をチェックする」必要があるが、これを行うと自己衝突の最大ボトルネックである「**GPUグローバルメモリアクセス（セル走査＋二分探索）**」を結局全員が実行することになり、高速化の恩恵がほぼ消滅する。
     - **結論**: 階層的BVHを持たない Uniform Grid 方式においては、**精度低下リスクとトレードオフが大きすぎるため、Taremin Cloth では**「**不採用（非推奨）**」とする。

### 1.2 Two-Level Bounding Sphere 階層的早期枝切り 【実装済み】
* **幾何学的理論**:
  * 頂点 $i$ と相手頂点 $j$ の距離二乗 $`d^2 = \|\mathbf{p}_i - \mathbf{p}_j\|^2`$ を内積（`dot` 1回）で評価。
  * 接触可能上界半径 $`R_{\text{bound}} = d_{\text{eff}} + (L_i + L_j) \times 1.3`$（実効厚み $`d_{\text{eff}}`$ = `effective_thick`、布の伸長に備え 1.3 倍マージン）を超えているペアは、トポロジー走査・V-V・V-T・E-E の全判定を即座にスキップ。
  * 相手三角形に対しても局所外接球 $`R_{VT} = d_{\text{eff}} + L_j \times 1.3 + \Delta \text{sweep}`$ により不要判定を早期遮断。
* **Taremin Cloth での実装効果**:
  * 80k頂点（約16万ポリゴン）において 86.60 ms (11.5 FPS) $`\to`$ 44.25 ms (22.6 FPS) と、所要時間を半減（約 1.96 倍高速化）しました。

---

## 2. 空間データ構造とGPU並列化 (Spatial Data Structures)

### 2.1 Sorted Uniform Grid (GPU Counting Sort方式) 【実装済み】
* **代表文献**:
  * **Green 2010**: *"Particle Simulation using CUDA"* (NVIDIA Whitepaper / GPU Gems)
  * **Blelloch 1990**: *"Prefix Sums and Their Applications"*
* **Taremin Cloth での実装**:
  * 従来の単方向リンクリスト（`atomicExchange` による動的ポインタチェイス）を完全撤廃。
  * ワークグループ共有メモリを用いた階層的排他 Prefix Sum（Blelloch Scan）とスキャッターにより、GPU内完結の並列 Counting Sort を実装。
  * セル走査を固定区間の連続メモリ読み出し（`cell_starts[h]..cell_starts[h+1]`）に刷新し、メモリアクセスの完全コアレッシング化を達成。100k頂点自己衝突ON時で 30.8 FPS を達成。

### 2.2 GPU Linear BVH (LBVH: Morton Codeによる階層化)
* **代表文献**:
  * **Karras 2012**: *"Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees"* (High Performance Graphics 2012)
    * [DOI: 10.2312/EGGH/HPG12/033-037](https://doi.org/10.2312/EGGH/HPG12/033-037)
* **特徴と適用性**:
  * 頂点・三角形のAABB中心を 32bit / 64bit Morton Code（Z-Order Curve）に変換し、GPU Radix Sort を経て階層ツリーを並列構築。
  * **Uniform Grid との比較**: ポリゴンサイズが均一な単一メッシュでは Uniform Grid の方が高速ですが、フリルやボタンなどの「極小メッシュ」と「巨大なスカート面」が混在する不均一メッシュや、多数の多重衣服レイヤー間の衝突判定で威力を発揮します。

---

## 3. 接触パイプラインとキャッシュ最適化 (Contact Pipeline & Caching)

### 3.1 接触候補ペアキャッシュ (I-Cloth 2018: Active Pair Caching)
* **代表文献**:
  * **Tang, Wang, Liu, Tong, Manocha 2018**: *"I-Cloth: Incremental Collision Handling for GPU-Based Interactive Cloth Simulation"* (ACM Transactions on Graphics / SIGGRAPH Asia 2018)
    * [DOI: 10.1145/3272127.3275005](https://doi.org/10.1145/3272127.3275005)
* **手法の概要**:
  * 広域探索（ブロードフェーズ: 空間ハッシュやBVH走査）と、厳密解決（ナローフェーズ: V-T, E-E, CCD交差判定）を時間軸上でデカップリング。
  * ブロードフェーズで発見された「実際に近接しているペア（Active Collision Pairs）」のみを GPU 上のコンパクトな配列（接触ペアバッファ）にアトミック追加・キャッシュ化。
  * 毎サブステップの自己衝突解決ループでは、全頂点スレッド（数万〜10万）から空間グリッドを27セル走査するのではなく、**キャッシュされた接触ペアリスト（数百〜数千個）に対してのみ専用スレッドを起動して並列解決**。
  * 接触ペアリストは数フレームに1回（または移動量が閾値を超えた時のみ）インクリメンタルに更新。
* **Taremin Cloth への適用性とトレードオフ**:
  * **利点**: 高密度メッシュにおいて、接触していない数万頂点が毎サブステップ空間セルを探索する無駄が完全に消滅し、ナローフェーズのGPUディスパッチが数ミリ秒以下へ短縮。
  * **課題**: GPU上での可変長ペアバッファ管理（アトミックカウンターによる領域確保）と、高速移動時のペア漏れ防止のための安全マージン（Enlarged AABB）の調整が必要。

### 3.2 サブステップ・デカップリング (Substep Decoupling) 【実装済み】
* **理論背景**:
  * 高周波の弾性振動（距離・曲げ）に対し、接触境界の変化は数ミリ秒スケールでは準静的である性質を利用。
  * 自己衝突のディスパッチ頻度を間引き（隔サブステップ実行等）、最終サブステップでのみ100%確実に自己衝突とPost-Relaxationを実行してフレーム出力時の貫通を完全に防止。
  * 実測において、100k頂点で 30.8 FPS $`\to`$ **41.6 FPS**（自己衝突オーバーヘッド -33% 削減）を達成。

---

## 4. XPBD拘束収束・安定化技術 (XPBD Convergence & Stability)

### 4.1 Chebyshev Acceleration (過大緩和による反復収束加速)
* **代表文献**:
  * **Wang 2015**: *"A Chebyshev semi-iterative approach for accelerating projective and position-based dynamics"* (ACM TOG / SIGGRAPH Asia 2015)
    * [DOI: 10.1145/2816795.2818063](https://doi.org/10.1145/2816795.2818063)
  * **Macklin et al. 2020**: *"Primal/Dual Descent Methods for Dynamics"* (Computer Graphics Forum 2020)
* **特徴と適用性**:
  * ガウス・ザイデル／ヤコビ型の拘束投影ループにおいて、前イテレーションの変位 $`\Delta \mathbf{x}_{k-1}`$ に対し、Chebyshev多項式の根に基づく過大緩和係数 $`\omega_k \in [1.0, 2.0)`$ を適用する半反復法。
  * 反復回数（`solver_iterations`）を増やすことなく、拘束の伝播速度を上げ、布の異常な伸びを約2〜3倍抑制。追加計算コストは実質ゼロ。

### 4.2 Long Range Attachments (LRA: 長距離付着拘束による伸び完全防止)
* **代表文献**:
  * **Kim et al. 2012**: *"Long range attachments - a method to simulate inextensible clothing in computer games"* (ACM SIGGRAPH / Eurographics SCA 2012)
    * [DOI: 10.2312/SCA/SCA12/305-310](https://doi.org/10.2312/SCA/SCA12/305-310)
  * **Müller et al. 2014**: *"Strain Based Dynamics"* (ACM SIGGRAPH / Eurographics SCA 2014)
* **特徴と適用性**:
  * ピン留め頂点から各布頂点までの測地線距離（布表面に沿った最大許容距離 $`D_{\text{geo}}`$）を初期化時に事前計算し、サブステップ終端で球体不等式拘束として投影。
  * 高密度メッシュで重力や激しい運動加速度が加わった際、布がゴムのように伸びて垂れ下がる現象を、たった1回のディスパッチで100%確実に防止。

### 4.3 Barrier Potential Contact (平滑バリア接触ポテンシャル)
* **代表文献**:
  * **Li et al. 2020**: *"Incremental Potential Contact (IPC)"* (ACM TOG / SIGGRAPH 2020)
    * [DOI: 10.1145/3386569.3392425](https://doi.org/10.1145/3386569.3392425)
* **特徴と適用性**:
  * 離散ステップでの急峻な押し出し変位に代わり、接触境界へ近づくにつれて対数的に反発力が高まる平滑バリア関数をXPBDのコンプライアンス拘束として定式化。
  * 接触面でのビリつき（ジッター）を低減し、シワの折り畳み部分が滑らかに滑り合う接触挙動を実現。

---

## 5. サーベイ技術一覧と総合比較マトリクス

| 技術分類 | 技術名 | 状態 / 文献 | 主な効果 | 計算コスト | 実装難易度 | Taremin Cloth 適合優先度 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **収束加速** | **Chebyshev過大緩和** | 検討中 (Wang 2015, Macklin 2020) | 反復数を増やさずに布の伸びを約2〜3倍抑制 | ほぼゼロ | ★☆☆☆☆ | **第1位 (即効性・コストゼロ・高剛性)** |
| **伸び防止** | **Long Range Attachments** | 検討中 (Kim 2012, Müller 2014) | 重力・激しい動きによる垂れ下がりを100%阻止 | 極小（~0.1ms） | ★★☆☆☆ | **第2位 (衣装のシルエット完全維持)** |
| **接触キャッシュ** | **接触候補ペアキャッシュ** | 検討中 (I-Cloth 2018) | 接近ペアのみをGPUバッファに収集して解く | 極小（ペア数依存） | ★★★☆☆ | **第3位 (自己衝突の更なる高速化)** |
| **接触平滑** | **Barrier Contact (IPC風)** | 検討中 (Li 2020) | 接触ジッター防止・滑らかなシワ滑り | ほぼゼロ | ★★★☆☆ | **第4位 (挟み込みの品質重視)** |
| **空間構造** | **GPU Linear BVH (LBVH)** | 検討中 (Karras 2012) | Morton Code階層化・不均一メッシュ対応 | 中（構築数ms） | ★★★★☆ | **第5位 (将来の多重衣装レイヤー時)** |
| **衝突枝切り** | **離散曲率事前カリング** | **不採用・非推奨** (Provot 1997, Volino 1994) | 平坦領域の衝突探索スキップ | 極小 | ★★☆☆☆ | **❌ 不採用 (スカート等のすり抜け破綻)** |
| **幾何枝切り** | **Two-Level Bounding Sphere** | **実装済み (フェーズ1)** | 広域外接球による非接触ペア早期枝切り | 内積1回 | ★☆☆☆☆ | **稼働中（精度100%維持・所要時間半減）** |
| **空間構造** | **Sorted Uniform Grid** | **実装済み (フェーズ2)** | GPU Counting Sortによる完全コアレッシング | 極小（Blelloch Scan） | ★★★☆☆ | **稼働中（精度100%維持・30 FPS達成）** |
| **時間刻み** | **サブステップ・デカップリング** | **実装済み (フェーズ3)** | 自己衝突ディスパッチ間引きと最終保証 | 削減（~33%短縮） | ★☆☆☆☆ | **稼働中（精度100%維持・41.6 FPS達成）** |
