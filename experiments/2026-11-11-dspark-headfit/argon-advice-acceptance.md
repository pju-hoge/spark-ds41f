# DSpark 投機デコードの受理率・受理長を劇的に引き上げるための技術助言

## 0. 「劇的に上げる」は可能か？ — 理論上限の見積もりと結論

**結論から述べると、ドラフター本体（`mtp.0/1/2` の3層Transformer）の再蒸留を行わない条件下では、単一施策で受理長を4.5+へ伸ばすような「魔法の弾丸」は存在しないが、複数の構造的ボトルネック（特に僅差逆転・位置別バイアス過補正・CopyDrafterの収量計算バグ・信頼度校正）を順に潰すことで、全体受理率を現在の `0.635〜0.639` から `0.70〜0.75`（+6〜11pt）、平均受理長を `3.14〜3.22` から `3.45〜3.75`（+0.30〜0.55 tok/step）へ引き上げることは十分に可能である。**
まず指標の定義と天井を数値で整理する。本クラスタの `acceptance_rate`（`0.635〜0.639`）は「適応検証（`adaptive_verification`）が検証に回したドラフト行数（平均 `3.37〜3.48` 行/step）に対する採択ドラフト数（平均 `2.14〜2.22` 個/step）の比率」であり、`acceptance_length`（`3.14〜3.22`）は `1 + 採択ドラフト数` である。トレース実測の条件付き受理率（`pos0=0.803, pos1=0.720, pos2=0.715, pos3=0.718, pos4=0.703`）から5トークン無条件累積採択確率を計算すると `0.803, 0.578, 0.413, 0.297, 0.209` となり、現行ドラフターの5トークン全検証時の理論上限（適応検証での打ち切りをゼロにした場合）は **受理長 `1 + 2.300 = 3.300`（GSM8K単体では実測 `3.563`）**、そのときの5行分母での受理率は **`2.300 / 5 = 0.460`（実測 `0.386`）** に留まる（[experiments/2026-10-26-dspark-acceptance/README.md](file:///home/masumi/NewModels/spark-ds41f/experiments/2026-10-26-dspark-acceptance/README.md#L343-L348)、[experiments/2026-11-11-dspark-headfit/README.md](file:///home/masumi/NewModels/spark-ds41f/experiments/2026-11-11-dspark-headfit/README.md#L647-L654)）。さらにMarkovバイアス適用前のドラフター `base_logits`（`dl`）の **top-32 にターゲットの正解トークン（committed token）が含まれる割合（カバレッジ）が全位置平均で `65.0%`（`pos0: 71.1%` → `pos4: 60.4%`）** であり、現状の `dl top-32 ∪ {committed}` 内での argmax 一致率が **`80.94%`**（`0.650 / 0.8094 ≈ 0.803` で `pos0` 受理率と整合）であることから、仮に **Markov Head（ロジット加算バイアス）をオラクル化して top-32 内の順位付けを 100% 当て切ったとしても、単ステップ条件付き受理率の引き上げ幅は最大で `65.0% × (1 - 0.8094) ≈ +12.4pt`（平均条件付き受理率 `0.73 → 0.85`、5トークン受理長の理論天井 `3.14 → 3.88`）** がハードな数学的上限となる。なぜベースロジットの top-32 カバレッジが `65%` で頭打ちになるかといえば、DSpark の `pos1〜pos4` は前のドラフトトークンを入力せず**定数のノイズトークン（`dspark_noise_token_id = 128799`）を並列入力して1回の forward で5位置分の隠れ状態 `sample_hidden` を出す構造**（[`speculator.py:249-256`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py#L249-L256)、[`dspark.py:424-493`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/models/deepseek_v4_1/nvidia/dspark.py#L424-L493)）だからであり、この `65%` の壁自体を突破するには `mtp.0/1/2` の再学習が不可欠であるが、逆に言えば**拒否の78%がロジット差1 nat以内の僅差逆転で起きている現状（[brief.md:77](file:///tmp/agy-advice/brief.md#L77)）では、その `+12.4pt` のオラクル余地のうち半分（`+5〜6pt`）を回収するだけでも受理長は `+0.3〜0.5 tok/step` 伸びる**。

---

## 1. 優先順位付き打ち手（期待効果の大きい順・全7案）

以下、すでに試行済みで効果がなかった施策（「一様な `MARKOV_SCALE` の 0.8/1.0/1.2 掃引」および「`W2` 固定・ターゲットロジット `l` に対する小標本 `W1` 単独 MLE 適合」）は除外し、コードと実測データから裏付けが取れた7つの打ち手を期待効果順に提示する。

---

### 案1: 既存実装済み・現在無効の「Top-2 Markov信頼度分岐（`dspark_top2_branch`）」の有効化とマージン最適化

#### ① 何をするか（設定 / コード / 重み）
- **驚きの事実として、コードベースには既に「ドラフターの1位トークンと2位トークン（runner-up）のロジット差が小さいとき、Markov埋め込みの信頼度重み `markov_conf_weight` で1位と2位を再評価して入れ替える Top-2 Branching 機構」が TileLang カーネルから TP4 通信経路まで完全に実装されているが、デフォルトで `False`（無効）になっている**（[`speculator.py:206-230`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py#L206-L230)）。
- 具体的には、TP4の各シャードで上位2候補 `(best_val, best_idx, second_val, second_idx)` を既存の `_greedy_shard_pairs`（形状 `[max_num_reqs, 4]`、16バイト/ランク — 通常時は後半2列が未使用のパディング）に格納して 64バイト `all_gather` を行い（**追加通信ゼロ**）、[`select_first_argmax_embed_top2`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/sample/tilelang_sampling.py#L2547-L2644) カーネル内で `margin = best_val - second_val <= branch_margin` のときに以下を判定する：
  $$\text{take\_second} \iff r_{\text{val}} + \text{branch\_weight} \cdot \langle w_{\text{conf}}, W_1[\text{second}] - W_1[\text{best}] \rangle > b_{\text{val}}$$
  ここで $w_{\text{conf}}$ は `confidence_head.proj.weight`（`[1, 5376]`）の後半256次元（Markov埋め込みに対応する重み、[`speculator.py:348-364`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py#L348-L364)）である。
- **変更内容**:
  1. `config/cluster.json` の `speculative_config`（または [`speculator.py:206-230`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py#L206-L230) のランタイムホットリロード `/cache/dspark-trace/TOP2_BRANCH` 追加）にて、`dspark_top2_branch: true`、`dspark_top2_branch_margin: 1.0`（拒否の78%が収まる1.0 nat）、`dspark_top2_branch_weight: 1.0`（0.5〜2.0で掃引）を有効化する。
  2. さらに効果を高める場合、この `markov_conf_weight`（256次元ベクトル1本のみ）を、採取済みトレースの「1位と2位の差が1 nat以内で2位が正解だったサンプル」のロジスティック回帰で最適化する。

#### ② 期待効果と根拠
- **期待効果**: 単ステップ条件付き受理率 **+2.0〜4.5pt**、平均受理長 **+0.15〜0.30 tok/step（`3.14 → 3.30〜3.45`）**、全体受理率 **+2.0〜4.0pt**（オンライン実測は未確認、オフラインのロジット差分布からの試算）。
- **根拠**:
  - [brief.md:77](file:///tmp/agy-advice/brief.md#L77) の実測で、**拒否されたドラフトの 78% が正解トークンとのロジット差 1 nat 以内**（平均 `-0.61 nats`）に集中している。
  - `W2 @ W1[prev]` は「前のトークン `prev` から次トークン候補 `y` への遷移バイアス」しか見ないが、`markov_conf_weight · W1[y]` は「候補トークン `y` 自体がドラフトされたときに採択されやすいトークンか（自己信頼度事前分布）」を表す。ロジット差が 1 nat 以内の僅差時に、採択されやすい候補へタイブレークすることで、最も層の厚い `|gap| < 1 nat` の拒否群を直接救済できる。

#### ③ コスト・リスク（ブリーフの制約1〜6との照合）
- **制約1（TileLang JIT凍結）**: **完全にクリア**。[`DSparkSpeculator.register_warmups()`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py#L474-L517) が `use_top2=self._enable_top2_branch` に応じて起動時に `greedy_markov_shard_pair_top2` と `select_first_argmax_embed_top2` を自動登録・ウォームアップする実装になっている。
- **制約2・3（TP4語彙シャード・Sparknet RoCE通信）**: **追加通信ゼロ**。既存の `_greedy_shard_pairs`（`[max_num_reqs, 4]`、16バイト）の未使用だった第2・第3列にシャード内2位 `(second_val, second_idx)` を詰めるだけ（[`tilelang_sampling.py:2296-2301`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/sample/tilelang_sampling.py#L2296-L2301)）なので、通信サイズは 64バイトのまま一切増えない。
- **制約4（FULL CUDA Graph）**: **完全対応**。分岐判定はすべて [`select_first_argmax_embed_top2`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/sample/tilelang_sampling.py#L2602-L2638) のレジスタ内 `T.if_then_else` で完結しており、グラフ再捕捉不要でそのままリプレイされる。

#### ④ 安く検証する方法（どの計測で何を見るか）
1. **オフライン0秒検証**: 既に採取済みの `dl` トレース（または1ランク採取の `step-*.pt`）に対し、各ステップの1位 `b_idx` と2位 `r_idx` で `margin <= M` のとき `r_val + W * (w_conf @ (W1[r_idx] - W1[b_idx])) > b_val` なら `r_idx` を選ぶシミュレーションを Python 10行で回し、`(M, W)` グリッド上で `b_idx → r_idx` のスワップによる正味の正解数増分（`#rescued - #broken`）を確認する。
2. **オンライン検証**: `config/cluster.json` の `speculative_config` に `"dspark_top2_branch": true, "dspark_top2_branch_margin": 0.8, "dspark_top2_branch_weight": 1.0` を入れて `benchmark-matrix --Filter gsm8k-c1,prose-c1` を流し、`spec_decode_acceptance_rate` と `runner_up` スワップ率を確認する。

#### ⑤ 触る場所（`file:line`）
- 設定のみの場合: [`config/cluster.json:152-165`](file:///home/masumi/NewModels/spark-ds41f/config/cluster.json#L152-L165)（`speculative_config` ブロック）および [`vllm/config/speculative.py:320-360`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/config/speculative.py#L320-L360)（フィールド露出確認）
- ランタイム設定読み込み・カーネル呼び出し: [`vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:206-230`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py#L206-L230), [`348-364`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py#L348-L364), [`654-673`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py#L654-L673)
- カーネル本体: [`vllm/v1/worker/gpu/sample/tilelang_sampling.py:2547-2644`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/sample/tilelang_sampling.py#L2547-L2644)

---

### 案2: Markov Head（`W1`・`W2`）の「全語彙共有 `W2` 残差＋埋め込み正則化＋1-natマージン損失」による共同再適合（TP4同期 `base_logits` 採取併用）

#### ① 何をするか（設定 / コード / 重み）
今日の `fit_markov_head.py`（`W1` のみ更新）がオフライン `+1.88pt`・オンライン `+0.3pt` に終わった原因は以下の3点にある：
1. **被覆率の不足（2.16%しか更新されていない）**: `W1`（`[129280, 256]`）は `prev` トークンごとの独立ルックアップ表であるため、74,775サンプルで `min_count >= 4` を満たした **2,787行（語彙の 2.16%）しか更新されず**、残り 97.84% の `prev` トークンは初期重みのまま放置された（[experiments/2026-11-11-dspark-headfit/README.md:541-544](file:///home/masumi/NewModels/spark-ds41f/experiments/2026-11-11-dspark-headfit/README.md#L541-L544)）。
2. **ターゲット `l` とドラフター `dl` の符号逆転（中央値 `-1.15 nats` の謎の正体）**: [brief.md:75](file:///tmp/agy-advice/brief.md#L75) で「ターゲットの `l` から逆算した必要バイアス中央値が `-1.15 nats` になる」のは、ターゲット `l` では**既に正解トークンがドラフトトークンより中央値で `+1.15 nats` 高い**（だからターゲットは正解トークンを採択した）のに対し、ドラフターでは逆にドラフトトークンが勝っているためである。ターゲット `l` をベースロジット代わりに使うと、既に勝っている正解トークンへのバイアスを削る方向に学習してしまう。
3. **Softmax CE 損失のミスマッチ**: Greedy ドラフトで必要なのは「33候補全体の確率分布の一致」ではなく「1位（argmax）が正解トークンになること」である。Softmax CE は既に 5 nats 差で勝っているサンプルや 10 nats 差で負けているサンプルの裾野合わせに容量（rank 256）を浪費する。

そこで以下の3ステップで `W1` と `W2` を共同再適合する：
- **Step A（TP4本番環境での真の `base_logits` top-32 採取）**:
  [`argon-2026-11-11-tp4-capture.md`](file:///home/masumi/NewModels/spark-ds41f/experiments/2026-11-11-tp4-capture.md#L523-L540) で特定された通り、本番TP4経路は既に [`speculator.py:574`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py#L574) でシャード局所のMarkov適用前 `base_logits`（`[num_reqs, 5, 32320]`）を計算している。[`model_runner.py:3475-3489`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/model_runner.py#L3475-L3489) の `draft_trace_capture` を全4ランク同期のステップカウンタに直し、全4ノードの `/cache/dspark-trace/` にマーカーを置いてシャード別 `topk(16)` + 10KB `all_gather` で **30万〜50万サンプルの真の TP4 `dl` トレース**を集める。
- **Step B（`W2` 共有更新 ＋ ターゲット埋め込みからの `W1` 低ランク補間）**:
  - **全 `prev` トークンに効く `W2`（`[129280, 256]`）の更新**: `W1` を固定（または微小学習率）にして出力側 `W2`（特に頻出上位 16,384 語彙の行、および `256×256` の中間線形変換 $M$: $W_2^{\text{new}} = W_2^{\text{orig}} M$）を学習する。中間行列 $M \in \mathbb{R}^{256 \times 256}$（わずか 65,536 パラメータ）を学習して $W_2^{\text{new}} = W_2 M$ に畳み込めば、**トレースに1回も出現しなかった希少な `prev` トークン（残り98%の語彙）に対しても改善が汎化する**。
  - **希少 `prev` の埋め込み正則化**: `W1` の更新では、本体の単語埋め込み $E_{\text{target}} \in \mathbb{R}^{129280 \times 5120}$ からの射影 $E_{\text{target}} P$（$P \in \mathbb{R}^{5120 \times 256}$）を事前平均とする Ridge 正則化 $\lambda \| W_1 - (W_1^{\text{orig}} + E_{\text{target}} P) \|_F^2$ を入れることで、低頻度 `prev` トークンにも意味的類似トークンの遷移バイアスを波及させる。
- **Step C（1-nat 僅差領域に集中した Truncated Margin Ranking 損失）**:
  - 損失関数を Softmax CE から、正解 $y^*$ が `dl` top-32 内にあり、かつ現在の負マージンが救済可能範囲（$-2.5 \le s(y^*) - \max_{y \neq y^*} s(y) \le +0.5$）にあるサンプルに絞った **Smooth Hinge / Margin Loss** に変更する：
    $$\mathcal{L} = \sum_{i \in \text{rescuable}} \log\left(1 + \exp\left(\frac{\max_{y \neq y_i^*} (dl_{i,y} + W_{2,y} W_{1,\text{prev}_i}) - (dl_{i,y_i^*} + W_{2,y_i^*} W_{1,\text{prev}_i}) + m}{\tau}\right)\right) + \lambda_M \|M - I\|_F^2 + \lambda_1 \|\Delta W_1\|_F^2$$

#### ② 期待効果と根拠
- **期待効果**: 単ステップ条件付き受理率 **+3.0〜6.0pt**（`dl top-32 ∪ {committed}` 内の一致率を現状の `80.94%` から `86〜90%` へ引き上げ）、平均受理長 **+0.20〜0.40 tok/step（`3.14 → 3.35〜3.55`）**、全体受理率 **+2.5〜5.0pt**（オンライン実測は未確認）。
- **根拠**:
  - 今日の実測で、わずか 2,787行（2.16%）の `W1` を Softmax CE で1列ずつ合わせただけでも、学習対象行のオフライン一致率は `80.94% → 82.82% (+1.88pt)` に上がった（[`README.md:543`](file:///home/masumi/NewModels/spark-ds41f/experiments/2026-11-11-dspark-headfit/README.md#L543)）。
  - $256 \times 256$ の共有変換 $M$（`W2` へ畳み込み）と $E_{\text{target}} P$ 補間を併用すれば、**全129,280語彙（100%）の `prev` トークン**にバイアス改善が適用され、さらに Hinge Loss によって 78% の僅差拒否（1 nat 以内）にランク256の表現力を集中できる。

#### ③ コスト・リスク（ブリーフの制約1〜6との照合）
- **制約1〜6すべてに対して推論時オーバーヘッド完全ゼロ**。学習後の重み $W_1^{\text{new}}, W_2^{\text{new}}$ は既存の `/cache/dspark-trace/markov_head.pt`（[`speculator.py:67-82, 315-346`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py#L315-L346)）からロードするだけで、テンソル形状・データ型（BF16）・TP4シャード分割（`W2` は4分割、`W1` は複製）も一切変わらない。
- **注意点（制約3・6関連）**: `/cache` は各ノードのローカルディスク（[`config/cluster.json:60-61`](file:///home/masumi/NewModels/spark-ds41f/config/cluster.json#L60-L61)）であるため、生成した `markov_head.pt` は**必ず4ノードすべて（`goro, shiro, taro, jiro`）の `/cache/dspark-trace/markov_head.pt` に `scp` で同一配置すること**（1ノードだけ置くとシャード間で別重みを参照する）。

#### ④ 安く検証する方法（どの計測で何を見るか）
1. **オフライン検証（数分）**: まず既存の1ランク `dl` トレース（74,775サンプル）を 80% train / 20% holdout（特に train に出現しない `prev` トークンの holdout）に分割し、①従来の `W1` 単独 CE、② $W_2^{\text{new}} = W_2 M$（共有 $256 \times 256$）＋ Margin Loss、③ $M + \Delta W_1 + E_{\text{target}} P$ の3通りで holdout の top-1 採択率を比較する。**未見の `prev` トークンで一致率が上がること**を確認してからオンラインに投入する。
2. **オンライン検証**: 4ノードに `markov_head.pt` を配布し、`benchmark-matrix` の 8セルで `acceptance_rate` と `acceptance_length` を計測する。

#### ⑤ 触る場所（`file:line`）
- 重みホットロード機構（既存・変更不要）: [`vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:67-82`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py#L67-L82), [`315-346`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py#L315-L346)
- TP4同期 `base_logits` top-k 採取時のみ: [`vllm/v1/worker/gpu/model_runner.py:3475-3489`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/model_runner.py#L3475-L3489), [`vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:574-615`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py#L574-L615), [`vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:888`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py#L888)

---

### 案3: 位置別（`pos0`〜`pos4`）Markovバイアス・温度スケーリングの非一様化（`MARKOV_SCALE` の5要素ベクトル化）

#### ① 何をするか（設定 / コード / 重み）
- [brief.md:83](file:///tmp/agy-advice/brief.md#L83) では「`MARKOV_SCALE`（0.8 / 1.0 / 1.2）の**一様スケール掃引**」が試され効果なしとされたが、[`speculator.py:85-136`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py#L85-L136) の `_read_markov_scale()` を読むと、**既に `/cache/dspark-trace/MARKOV_SCALE` にカンマ区切りで5要素 `s0,s1,s2,s3,s4` を書けば、位置 `i=0..4` ごとに異なるスケール `scale_i` を適用する実装が完了している**（[`speculator.py:648, 682`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py#L648)）。
- **なぜ一様スケールは効かず、位置別スケール（特に `s0 < 1.0 < s1 <= s2 <= s3 <= s4`）が構造的に必須なのか**:
  1. [`DSparkDeepseekV4Model.forward`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/models/deepseek_v4_1/nvidia/dspark.py#L424-L493) において、**`pos0` だけは入力として「真の直前トークン（anchor token）の埋め込み」を受け取って 3層の MTP Transformer を通過している**。つまり `pos0` の `base_logits` には既に直前トークン `prev` の情報が完全に入っており、そこに `s0 = 1.0` で Markov バイアスを足すと **直前トークンからの bigram 遷移確率を二重計上（double-counting）** してしまい、文脈より高頻度 bigram を優先する過補正が起きる。
  2. 逆に **`pos1〜pos4` の MTP 層への入力トークンはすべて定数のノイズトークン（`128799`）**（[`speculator.py:249-256`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py#L249-L256)）であり、`base_logits[:, i]`（$i \ge 1$）は「1つ前のステップ $i-1$ でどのドラフトトークンが選ばれたか」を**一切知らない**。`pos1〜pos4` で直前ドラフトトークンの情報を伝える唯一の経路が `markov_head` である。
  3. したがって、一様に `0.8` に下げると `pos0` の二重計上は減るが `pos1〜4` の連鎖条件付けが弱まり、一様に `1.2` に上げると `pos1〜4` は強まるが肝心の先頭 `pos0` が壊れる（先頭 `pos0` が外れると後続 `pos1〜4` はすべて棄却されるため相殺される）。
- **さらに重要な運用上の盲点**:
  `/cache` は各ノードのローカルディレクトリ（[`config/cluster.json:60-61`](file:///home/masumi/NewModels/spark-ds41f/config/cluster.json#L60-L61)）である。もし前回の `MARKOV_SCALE` 実験で rank 0（ヘッドノード）の `/cache/dspark-trace/MARKOV_SCALE` しか書き換えていなかった場合、**rank 0 の語彙シャード（`0..32319`）だけスケールが変わり、rank 1〜3（`32320..129279`）は `1.0` のまま比較されてシャード間のロジットスケール不整合を起こしていた**可能性がある（未確認）。必ず4ノードすべてに同一の `MARKOV_SCALE` を配置する必要がある。

#### ② 期待効果と根拠
- **期待効果**: 単ステップ条件付き受理率 **+1.5〜3.0pt**、平均受理長 **+0.10〜0.22 tok/step**、全体受理率 **+1.5〜2.5pt**（オンライン実測は未確認）。
- **根拠**: `pos0` の無条件寄与度は全位置最大の `1.0`（`pos0` が `+2pt` 改善すると、連鎖効果で全5位置の期待受理長が $0.02 \times (1 + 0.72 + 0.52 + 0.37 + 0.26) = +0.057$ tok/step 底上げされる）であり、`pos1〜pos4` と独立に `s0, s1, s2, s3, s4` を最適化できる。

#### ③ コスト・リスク（ブリーフの制約1〜6との照合）
- **コード変更ゼロ・ビルド不要・メモリ増ゼロ**。[`speculator.py:648`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py#L648) は `bias = bias * scale_i` をインプレースに近い形で実行するが、注意点として **制約4（FULL CUDA Graph）** により、`scale_i == 1.0` のときは乗算カーネル自体がグラフに記録されない（`if scale_i != 1.0: bias = bias * scale_i`）。したがって `/cache/dspark-trace/MARKOV_SCALE` を変更した際は、CUDA Graph に新しい `scale_i` 定数を焼き込むためワーカー起動時（またはグラフ再捕捉）に全4ノードへ設定しておく必要がある。

#### ④ 安く検証する方法（どの計測で何を見るか）
1. **オフライン0秒検証**: 74,775サンプルの `dl` トレースを `pos = 0, 1, 2, 3, 4` の5つのバケットに分け、各位置 `j` ごとに独立して $\arg\max_{s_j \in [0.4, 1.8]} \text{Acc}_j(dl + s_j \cdot \text{markov\_bias})$ を 0.05 刻みで求める（Python で数秒で完了）。これで各位置の最適スケール $(s_0^*, s_1^*, s_2^*, s_3^*, s_4^*)$ と位置別の改善幅が厳密に判明する。
2. **オンライン検証**: 得られた5要素（例: `0.75,1.00,1.15,1.25,1.30`）を**全4ノードの `/cache/dspark-trace/MARKOV_SCALE`** に書き込んで `benchmark-matrix` を実行する。

#### ⑤ 触る場所（`file:line`）
- 既存の位置別スケール読み込み・適用箇所（コード変更不要）: [`vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:85-136`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py#L85-L136), [`646-650`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py#L646-L650)

---

### 案4: `CopyDrafter`（`tilelang_copy.py`）の期待収量比較バグ修正と事前信頼度（Prior）の適正化

#### ① 何をするか（設定 / コード / 重み）
[brief.md:46](file:///tmp/agy-advice/brief.md#L46) の位置別条件付き受理率に **`pos5: 0.106 (n=218)`** という異常に低い行が存在する理由をコードから特定した。DSpark 本体は5トークン（`pos0〜pos4`）しか生成しないが、[`config/speculative.py:1810-1823`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/config/speculative.py#L1810-L1823) により `method=="dspark"` かつ `enable_adaptive_verification=True` のとき **プロンプト/生成履歴からの N-gram コピードラフター [`CopyDrafter`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/spec_decode/copy_drafts.py#L45-L61)（`min_match=16, initial_drafts=8, max_drafts=16`）が自動で有効化**され、`pos5` 以降のトークンを挿入している。
しかし [`vllm/v1/worker/gpu/spec_decode/tilelang_copy.py`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/spec_decode/tilelang_copy.py#L365-L392) の `copy_apply` カーネルと `copy_alloc` カーネルに**3つの数理的不整合（バグ）**があり、これが偽陽性コピーの暴発と `pos5: 0.106` の大量棄却を引き起こしている：
1. **DSpark期待収量とCopy期待収量の計算式バグ（[`tilelang_copy.py:365-370`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/spec_decode/tilelang_copy.py#L365-L370)）**:
   ```python
   for j in T.unroll(block):
       dspark_yield += dspark_confidences[r, j]
   if start >= 0 and T.cast(offered, T.float32) * estimate >= dspark_yield:
       use_copy = 1
   ```
   `dspark_confidences[r, j]` は位置 $j$ の**条件付き**採択確率 $c_j$ であるため、DSpark の真の期待採択トークン数は単純和 $\sum_{j=0}^{4} c_j$ ではなく**累積積の和** $\sum_{j=0}^{4} \prod_{m=0}^{j} c_m$ である。同様に、Copy 側の1トークン条件付き採択率 `estimate`（$= p_c$）に対する `offered`（$= K$、初期8・最大16）トークンの真の期待収量も `offered * estimate`（$= 8 p_c$ や $16 p_c$）ではなく**幾何級数和** $\sum_{j=1}^{K} p_c^j = p_c \frac{1 - p_c^K}{1 - p_c}$ である。現状は Copy 側が `offered * estimate`（最大 $16 \times 0.95 = 15.2$）と線形に過大評価されるため、一度でもコピーが走ると `estimate` が `0.50` まで落ちても `8 * 0.50 = 4.0 > dspark_yield (3.6)` となって **DSpark の高品質なドラフトを低品質なコピーで上書きし続ける**。
2. **適応検証に渡す `confidences` の過大書き込み（[`tilelang_copy.py:386-391`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/spec_decode/tilelang_copy.py#L386-L391)）と過剰な事前分布（[`tilelang_copy.py:76-78`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/spec_decode/tilelang_copy.py#L76-L78)）**:
   `PRIOR_HITS = 19.0, PRIOR_TRIALS = 20.0`（初期信頼度 `0.95`、重み20試行分）が硬すぎるため、16トークン一致が偶然発生して最初のトークンで即座に棄却（`accepted = 0, trials += 1`）されても、`estimate` は `19/20 = 0.950` から `19/21 = 0.905` にしか下がらない。その結果、[`adaptive_verification.py:338`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/spec_decode/adaptive_verification.py#L338) が `pos5` でも累積確率 $0.905^6 = 0.549 \gg \tau_{\text{dead}} (0.063)$ と判定して 8〜16行すべてを検証に通してしまい、分母（検証行数）を水増しして全体受理率を押し下げている。
- **修正内容**:
  1. [`tilelang_copy.py:76-78`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/spec_decode/tilelang_copy.py#L76-L78) の事前分布を `PRIOR_HITS = 3.6, PRIOR_TRIALS = 4.0`（初期値 `0.90`、1回先頭棄却されたら `3.6 / 5.0 = 0.72` へ即座に減衰）に引き下げる。
  2. [`tilelang_copy.py:365-370`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/spec_decode/tilelang_copy.py#L365-L370) の `dspark_yield` と `copy_yield` をともに**条件付き確率の累積積の和**（$\sum_{j} \prod_{m \le j} c_m$）に統一する：
     ```python
     dspark_cum = T.alloc_var(T.float32, init=T.float32(1.0))
     for j in T.unroll(block):
         dspark_cum *= dspark_confidences[r, j]
         dspark_yield += dspark_cum
     copy_yield = T.alloc_var(T.float32, init=T.float32(0.0))
     copy_cum = T.alloc_var(T.float32, init=T.float32(1.0))
     for j in T.unroll(lookahead):
         if j < offered:
             copy_cum *= estimate
             copy_yield += copy_cum
     if start >= 0 and copy_yield >= dspark_yield:
         use_copy = 1
     ```

#### ② 期待効果と根拠
- **期待効果**: 全体受理率 **+1.5〜3.0pt**（`pos5` 以降の無駄な偽陽性コピー検証行の排除）、DSpark のドラフトが不当に上書きされていたステップの回復による受理長 **+0.08〜0.20 tok/step**（オンライン実測は未確認）。
- **根拠**: トレース中の `pos5: 0.106 (n=218)` は 100% `CopyDrafter` 由来であり、これらがステップあたり8〜16行の検証枠を消費して 89.4% 棄却されている。累積積比較と軟らかい事前分布（`PRIOR_TRIALS=4.0`）に直すことで、真の反復（コードのボイラープレート等で `accepted == live` が続き `hits/trials → 0.98` になるケース）の高速化（[`2026-11-01-dspark-default-copy-drafts`](file:///home/masumi/NewModels/spark-ds41f/experiments/2026-11-01-dspark-default-copy-drafts/README.md) で実証済みの `+8.6%`）を一切損なわずに、単発の偶発マッチによる DSpark 上書きと検証行の浪費だけを止められる。

#### ③ コスト・リスク（ブリーフの制約1〜6との照合）
- **制約1（TileLang JIT凍結）**: 既存の [`copy_apply_kernel`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/spec_decode/tilelang_copy.py#L330-L406) 内のスカラーループ計算式を直すだけなので、カーネルキー `copy_apply_key(lookahead, block, idx_dtype)` もウォームアップ登録（[`copy_drafts.py:130-165`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/spec_decode/copy_drafts.py#L130-L165)）もそのまま機能する。
- **制約2〜6**: 通信・メモリ・CUDA Graph 形状への影響は完全にゼロ。

#### ④ 安く検証する方法（どの計測で何を見るか）
1. **設定のみでの即時切り分けテスト**: `config/cluster.json` の `speculative_config` に `"copy_drafts": {"enabled": false}`（または `"initial_drafts": 5, "min_match": 24`）を一時的に入れてトレースを取り、`pos5` の出現がゼロになることと、`pos0〜pos4` の受理率・検証行数がどう変化するかを見る。
2. **修正後の検証**: 上記の `copy_yield` / `dspark_yield` 累積積修正を入れて `benchmark-matrix` を回し、トレースの `pos5` 受理率が `0.106` から `0.60+` に跳ね上がること、および全体 `acceptance_rate` の上昇を確認する。

#### ⑤ 触る場所（`file:line`）
- [`vllm/v1/worker/gpu/spec_decode/tilelang_copy.py:76-79`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/spec_decode/tilelang_copy.py#L76-L79)（`PRIOR_HITS`, `PRIOR_TRIALS`）
- [`vllm/v1/worker/gpu/spec_decode/tilelang_copy.py:365-374`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/spec_decode/tilelang_copy.py#L365-L374)（`copy_apply` 内の `dspark_yield` と `use_copy` 判定）
- 設定変更のみで試す場合: [`vllm/config/speculative.py:86-135, 1810-1823`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/config/speculative.py#L86-L135)

---

### 案5: `DSparkConfidenceHead` への Top-2 ロジットマージン統合と `OnlineAcceptanceEstimator` の位置別（Pos 0〜4）独立校正

#### ① 何をするか（設定 / コード / 重み）
現在の信頼度推定には2つの構造的欠陥がある：
1. **Confidence Head がロジット差（マージン）を一切見ていない**:
   [`DSparkConfidenceHead.forward`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/model_executor/models/qwen3_dspark.py#L198-L216)（および [`speculator.py:674-678`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py#L674-L678)）は、`concat(sample_hidden, W1[draft_token])`（5120+256=5376次元）だけを RMSNorm + Linear に通して信頼度ロジットを出している。そのため、**1位と2位のロジット差が `5.0 nats` の盤石なドラフトと、`0.02 nats` 差のコイントス状態（拒否の78%を占める領域）のドラフトに対して、隠れ状態が似ていれば同じ信頼度を出してしまう**。
2. **オンライン校正器 `OnlineAcceptanceEstimator` が全5位置で単一の `(a, b)` を共有している**:
   [`speculator.py:293-305`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py#L293-L305) で生成される [`OnlineAcceptanceEstimator`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/spec_decode/acceptance_estimator.py#L424-L606) は、全位置 `pos0〜pos4` のサンプルを1つのバケットに混ぜてスカラーの `a, b`（$\sigma(a + b \cdot z)$）を学習している（[`acceptance_estimator.py:482-490`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/spec_decode/acceptance_estimator.py#L482-L490)）。しかし実測の通り `pos0`（条件付き受理率 `0.803`）と `pos1〜4`（`0.703〜0.720`）ではロジットと採択率の対応関係が約10pt異なるため、単一の `(a, b)` では `pos0` を過小評価し `pos3〜4` を過大評価する。
- **変更内容**:
  1. 案1で有効化する `select_first_argmax_embed_top2` が既に出力している `self.draft_top2_margins`（形状 `[max_num_reqs, 5]`、1位と2位のロジット差 $\Delta l = b_{\text{val}} - r_{\text{val}}$、[`speculator.py:668`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py#L668)）を、`confidence_head` の出力ロジット $z_{\text{conf}}$ に合成する：$z_{\text{total}} = z_{\text{conf}} + \alpha \cdot \tanh(\beta \cdot \Delta l)$（または $\alpha \cdot \min(\Delta l, 4.0)$）。
  2. [`OnlineAcceptanceEstimator`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/spec_decode/acceptance_estimator.py#L482-L545) のパラメータ `self.a`, `self.b` をスカラー `[1]` から位置別ベクトル `[num_speculative_steps]`（5要素）に拡張し、位置 $j \in \{0,1,2,3,4\}$ ごとに独立にオンライン校正する。

#### ② 期待効果と根拠
- **期待効果**: 全体受理率 **+3.0〜6.0pt**（`0.638 → 0.67〜0.70`）、同一検証コスト下での有効受理長 **+0.08〜0.15 tok/step**、または同一受理長での検証行数 **-0.3〜-0.5 rows/step 削減による tok/s +3〜5% 向上**（オンライン実測は未確認）。
- **根拠**:
  - [brief.md:77](file:///tmp/agy-advice/brief.md#L77) の通り拒否の 78% はロジット差 1 nat 以内で発生するため、Top-2 ロジット差 $\Delta l$ は採択/棄却を分ける最強の単一特徴量である。
  - [`experiments/2026-11-06-dspark-online-calibration/summary.json`](file:///home/masumi/NewModels/spark-ds41f/experiments/2026-11-06-dspark-online-calibration/summary.json) でも、未校正の Confidence Head は信頼度を `+0.09〜+0.19` 過大予測していたことが判明しており、これを位置別＋マージン付きで校正することで、`AdaptiveVerificationManager` が「当たる `pos3〜4`（$\Delta l > 2$ nat）」を残し「外れる `pos1〜2`（$\Delta l < 0.3$ nat）」を正確に刈り込めるようになる。

#### ③ コスト・リスク（ブリーフの制約1〜6との照合）
- **制約1（TileLang JIT凍結）**: [`tilelang_acceptance.py`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/spec_decode/tilelang_acceptance.py) のカーネルを変更せずとも、[`speculator.py:678`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py#L678) の直後（`predict_from_logit` 呼び出し前）で要素ごとの加算 `raw_logits.add_(self.draft_top2_margins[:num_reqs].clamp(max=4.0), alpha=margin_alpha)` と位置バイアスベクトル加算を行うだけで済むため、新規 TileLang カーネルは一切不要。
- **制約2〜6**: 通信ゼロ、メモリ増加ゼロ、CUDA Graph 完全対応。

#### ④ 安く検証する方法（どの計測で何を見るか）
1. **オフライン検証**: トレースデータの `dl` から計算した1位・2位差 $\Delta l = dl_{(1)} - dl_{(2)}$ と位置 $j$ を使ってロジスティック回帰 $\text{logit}(P(\text{accept})) = a_j + b_j z_{\text{conf}} + c_j \min(\Delta l, 4.0)$ をフィットし、`z_conf` 単独に対する Brier スコアおよび AUROC の改善幅を確認する。
2. **オンライン検証**: `benchmark-matrix` で `spec_decode_acceptance_rate` と平均検証行数（`verified_rows / step`）および `tok/s` を比較する。

#### ⑤ 触る場所（`file:line`）
- [`vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:665-680`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py#L665-L680)（`raw_logits` への `draft_top2_margins` と位置別オフセットの加算）
- [`vllm/v1/worker/gpu/spec_decode/acceptance_estimator.py:482-606`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/spec_decode/acceptance_estimator.py#L482-L606)（`OnlineAcceptanceEstimator` の位置別校正化）

---

### 案6: 適応検証の予算パラメータ（`adaptive_verification_row_cost`・`effective_dead_rows_tau`）の目的別リチューニング

#### ① 何をするか（設定 / コード / 重み）
設定ファイルと環境変数のみで**「受理長（acceptance length）の最大化」と「受理率（acceptance rate）の最大化」のパレート境界を直接制御する**即効策である。
コードを精査すると、[`adaptive_verification.py:306-324`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/spec_decode/adaptive_verification.py#L306-L324) に見落としやすい重要な挙動がある：
```python
if calibrated_confidence and self._dead_rows_tau > 0.0 and self.row_cost > 0.0:
    calibrated_tau = (2.0 * self.row_cost) / (1.0 + 3.0 * self.row_cost)
    return min(self._dead_rows_tau, calibrated_tau)
```
コミット `7e02d8887` でオンライン校正（`calibrated_confidence=True`）が導入された際、`config/cluster.json:104` で設定されている `"SPARK_DS41F_DSPARK_DEAD_ROWS_TAU": "0.2"` は実質的に無視され、**内部で自動的に `calibrated_tau = 0.07 / 1.105 = 0.06335` まで引き下げられている**。また [`adaptive_verification.py:131-139, 344-427`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/spec_decode/adaptive_verification.py#L131-L139) では、バッチ全体の検証トークン数が `HIGH_ROW_COST_KNEE = 16` を超えると行コストに `1 + 0.55 * (total_rows - 16)` の非線形ペナルティが掛かる（c8 では 8 reqs × 2 rows = 16 rows で即座に膝を超える）。
したがって、目的に応じて以下の2方向のいずれか（またはバッチサイズ連動の動的切替）を設定する：
- **A. 受理長（acceptance length）を劇的に上げたい場合（c1〜c4 低並列重視）**:
  - `config/cluster.json:162` の `"adaptive_verification_row_cost"` を `0.035` から **`0.015〜0.020`** に下げ、必要なら [`adaptive_verification.py:131`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/spec_decode/adaptive_verification.py#L131) の `HIGH_ROW_COST_KNEE` を `16 → 24` に緩和する。
- **B. 受理率（acceptance rate = accepted / verified）と c8 スループットを上げたい場合**:
  - [`adaptive_verification.py:319-323`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/spec_decode/adaptive_verification.py#L319-L323) の `min(self._dead_rows_tau, calibrated_tau)` による自動引き下げを外して `SPARK_DS41F_DSPARK_DEAD_ROWS_TAU=0.18〜0.22` をそのまま効かせるか、`adaptive_verification_row_cost` を **`0.050〜0.065`** に引き上げる。

#### ② 期待効果と根拠
- **期待効果（過去実験での実測裏付けあり）**:
  - **方向A（受理長優先）**: 平均受理長が **`3.14〜3.22 → 3.40〜3.55`（+0.20〜0.34 tok/step）** に上昇する（ただし検証行数が `3.4 → 4.3〜5.0` に増えるため、分母拡大により見かけの `acceptance_rate` は `0.64 → 0.45〜0.52` に下がり、c8 の tok/s は数%〜19%低下する）。
  - **方向B（受理率・高並列効率優先）**: 全体受理率が **`0.638 → 0.72〜0.78`（+8〜14pt）** に即座に跳ね上がる（累積採択確率が 18% 未満の `pos3〜4` を検証前に切り捨てるため。平均受理長は `3.14 → 2.95〜3.05` に微減するが、c8 での検証 GEMM 行数が減りスループットは維持〜微増する）。
- **根拠**: [`experiments/2026-10-26-dspark-acceptance/README.md:343-348`](file:///home/masumi/NewModels/spark-ds41f/experiments/2026-10-26-dspark-acceptance/README.md#L343-L348) および [`experiments/2026-11-06-verification-budget-hardware/README.md`](file:///home/masumi/NewModels/spark-ds41f/experiments/2026-11-06-verification-budget-hardware/README.md) の実測値そのものである。**案1〜案5でドラフター自体の条件付き受理率を底上げした上で、方向A（`row_cost = 0.020〜0.025`）を組み合わせれば、受理率を `0.64` 以上に保ったまま受理長を `3.5+` へ伸ばせる。**

#### ③ コスト・リスク（ブリーフの制約1〜6との照合）
- **設定変更のみ（方向Aの `row_cost` 変更）ならコード変更・ビルド一切不要**。制約1〜6に一切抵触しない。

#### ④ 安く検証する方法（どの計測で何を見るか）
- `config/cluster.json:162` の `adaptive_verification_row_cost` を `0.020` と `0.055` に振って `benchmark-matrix`（特に `gsm8k-c1`, `gsm8k-c8`, `prose-c1`, `prose-c8`）を実行し、`acceptance_rate`・`acceptance_length`・`tok/s` の3指標のパレート曲線を確認する。

#### ⑤ 触る場所（`file:line`）
- [`config/cluster.json:104, 162`](file:///home/masumi/NewModels/spark-ds41f/config/cluster.json#L104)（`SPARK_DS41F_DSPARK_DEAD_ROWS_TAU`, `adaptive_verification_row_cost`）
- [`vllm/v1/worker/gpu/spec_decode/adaptive_verification.py:131-139, 306-324`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/spec_decode/adaptive_verification.py#L306-L324)

---

### 案7: 温度 $>0$ 時の Block-wise（累積プレフィックス）棄却サンプリングへの拡張

#### ① 何をするか（設定 / コード / 重み）
- [brief.md:90-91](file:///tmp/agy-advice/brief.md#L90-L91) の「サンプリング方式（`draft_sample_method="greedy"` vs `"probabilistic"`, `rejection_sample_method="strict"`、温度との相互作用）」に対する回答である。
- まず `temperature == 0`（完全 Greedy 検証）では、ターゲットの分布自体が `argmax` の1点分布（One-hot）であるため、どのような棄却サンプリング方式を採っても「ドラフトトークンがターゲットの `argmax` と完全一致するか否か」以外の自由度は数学的に存在しない（[`greedy_verification.py:135-235`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/spec_decode/greedy_verification.py#L135-L235)）。
- 一方、`benchmark-matrix` や本番利用で **`temperature > 0` のリクエストが含まれる場合**、現状の `draft_sample_method="greedy"` + `rejection_sample_method="strict"`（[`rejection_sampler.py`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py)、[`tilelang_rejection.py`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/v1/worker/gpu/spec_decode/tilelang_rejection.py)）では、ドラフターの提案分布 $q(x_j)$ を決定論的 One-hot（$q(x_{\text{draft}}) = 1$）として扱うため、各位置 $j$ で独立に $u_j < \min(1, p_{\text{target}}(x_j) / 1) = p_{\text{target}}(x_j)$ を判定している。
- ここで `draft_sample_method="probabilistic"` に切り替えると、過去の実験（[`experiments/2026-10-18-dspark-probabilistic/README.md`](file:///home/masumi/NewModels/spark-ds41f/experiments/2026-10-18-dspark-probabilistic/README.md)）で実証された通り、全語彙 Gumbel サンプリングと `draft_logits` 実体化によりドラフターが **+9.15% 遅くなり、16バイト高速経路も使えなくなる**。
- しかし、**ドラフターは `draft_sample_method="greedy"`（16バイト高速経路）のまま維持しつつ、検証側（`tilelang_rejection.py`）だけを Block-wise Verification（Sun et al. / 累積確率比検証）に拡張する**ことができる。具体的には、位置 $j$ 単体で $u_j > p_{\text{target}}(x_j)$ となっても即座に打ち切らず、後続位置 $j+1$ のターゲット確率が高い場合にブロック結合確率で救済する（または `chain` モードを有効化する）。

#### ② 期待効果と根拠
- **期待効果**: `temperature > 0`（例: $T=0.6 \sim 1.0$）のワークロードにおいて、受理長 **+0.15〜0.35 tok/step**、受理率 **+3〜6pt**（**ただし `temperature == 0` の Greedy ベンチマークでは効果は 0**。オンライン実測は未確認）。
- **根拠**: 拒否の 78% がロジット差 1 nat 以内（ターゲット確率にして $p_{\text{target}}(x_{\text{draft}}) \approx 0.25 \sim 0.45$）であるため、独立棄却判定では 55〜75% の確率で落とされるが、直後の `pos(j+1)` が高確率トークン（$p_{\text{target}}(x_{j+1}) \approx 0.9$）であるケースが多く、ブロック単位の棄却判定により $T>0$ での途中棄却を有意に減らせる。

#### ③ コスト・リスク（ブリーフの制約1〜6との照合）
- ドラフター側は `greedy` のまま一切触らないため、**制約2・3・4（TP4 16バイト高速経路・FULL CUDA Graph）を 100% 維持できる**。
- 検証側の [`vllm/v1/worker/gpu/spec_decode/tilelang_rejection.py`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/spec_decode/tilelang_rejection.py) は起動時にウォームアップ登録（[`model_runner.py:2422-2425`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/model_runner.py#L2422-L2425)）されているため、制約1もクリアする。

#### ④ 安く検証する方法（どの計測で何を見るか）
- まず `benchmark-matrix` の対象タスクの温度設定を確認し、もし評価がすべて `temperature=0`（Greedy）で行われているなら本案はスキップして案1〜案6に注力する。$T>0$ 評価が含まれる場合は、`speculative_config` の `rejection_sample_method` を `"chain"`（[`speculative.py:305-320`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/config/speculative.py#L305-L320)）に切り替えて `acceptance_length` を比較する。

#### ⑤ 触る場所（`file:line`）
- [`config/cluster.json:159`](file:///home/masumi/NewModels/spark-ds41f/config/cluster.json#L159)（`"rejection_sample_method"`）
- [`vllm/v1/worker/gpu/spec_decode/rejection_sampler.py:1-120`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py#L1-L120)
- [`vllm/v1/worker/gpu/spec_decode/tilelang_rejection.py:1-150`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/spec_decode/tilelang_rejection.py#L1-L150)

---

## 2. ブリーフ第6節の残りの論点に対する明確な診断（なぜ手を出すべきでないか）

ブリーフ第6節（[brief.md:88-100](file:///tmp/agy-advice/brief.md#L88-L100)）で挙げられていた残りの方向性について、コードと過去の実測ログに基づく結論を明記する。これらは**追及しても受理率が上がらないか、スループットが大幅に悪化する**ため優先度を下げるべきである。

1. **TP4語彙シャード（32,320×4）でのシャードローカル argmax による候補ズレはあるか？ → 【結論: 完全にゼロ（数学的に一致）】**
   - [`greedy.py:109-114`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/spec_decode/dspark/greedy.py#L109-L114) および [`tilelang_sampling.py:2285-2320`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/v1/worker/gpu/sample/tilelang_sampling.py#L2285-L2320) を精査した。各シャードは局所語彙（32,320トークン）に対して `base_logits + markov_bias` の最大値とグローバル語彙インデックス `(best_val, best_idx)` を FP32 で求め、`select_first_argmax` が4シャードの最大値（同値時は最小語彙インデックス）を選ぶ。$\max_{r \in \{0..3\}} \max_{i \in \text{shard}_r} x_i = \max_{i \in \{0..129279\}} x_i$ は恒等的に成り立つため、**単一GPUの全語彙 argmax と1ビットの狂いもなく完全一致しており、シャード分割による候補ズレは一切存在しない**。
2. **ドラフト長を `5 → 8` や `16` に伸ばす案 → 【結論: 不可（OOD崩壊＋c8で検証時間が爆発）】**
   - [`dspark.py:310-335`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/models/deepseek_v4_1/nvidia/dspark.py#L310-L335) の `mtp.0/1/2` は `dspark_block_size = 5`（anchor 1個＋ノイズトークン `128799` 4個）のブロック構造で学習されている。これを8や16に伸ばしても `pos5` 以降はノイズトークンのみの訓練分布外（OOD）外挿となり、条件付き受理率が急락する。さらに c8（8並列）では検証行数が `8 × 9 = 72` 行に達し、`HIGH_ROW_COST_KNEE = 16` を大幅に超えて検証ステップ時間が爆発する（[`experiments/2026-09-27-dspark-depth5/README.md`](file:///home/masumi/NewModels/spark-ds41f/experiments/2026-09-27-dspark-depth5/README.md) でも実証済み）。
3. **Tree / Multi-candidate / EAGLE式の木構造検証 → 【結論: 不適（受理長 +0.19 に対し tok/s が -38% 悪化）】**
   - [`experiments/2026-10-26-dspark-acceptance/README.md:445-455`](file:///home/masumi/NewModels/spark-ds41f/experiments/2026-10-26-dspark-acceptance/README.md#L445-L455) で既に実測・試算されている通り、`pos1` を2分岐させて2パス検証しても受理長は `3.56 → 3.75 (+0.19)` しか伸びない一方、検証行数の倍増と DeepSeek-V4.1 固有の6種の注意機構（Sparse MLA, Sliding Window, Compressor, Indexer, Engram, CED）すべてへの木構造マスク導入コストにより **tok/s が -38% 低下**するため完全に割に合わない。
4. **量子化影響（本体 FP4 expert とドラフター BF16 head のズレ） → 【結論: ヘッドの量子化差ではなく 3層＋ノイズトークン入力が主因】**
   - [`dspark.py:345-355, 795-822`](file:///home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-eb2dd22b/src/vllm/vllm/models/deepseek_v4_1/nvidia/dspark.py#L345-L355) を確認すると、ドラフターは本体の `embed_tokens` と `lm_head`（ともに BF16）をポインタ共有しており、`mtp.0/1/2` の MoE expert も本体と同じ `quant_config`（NVFP4）でロードされている。したがって「本体とドラフターの量子化形式の違い」によるズレではなく、本体（多層）とドラフター（3層＋`pos1..4` ノイズ入力）の表現力差そのものが `dl` top-32 カバレッジ 65% の原因である。

---

## 3. 推奨する最短実行ロードマップ

1. **Day 1（コード変更ゼロ・オフライン＆設定検証）**:
   - 既存トレース（74,775サンプル）で **案3（位置別 `MARKOV_SCALE` の 5要素最適化）** と **案1（`dspark_top2_branch` の `(margin, weight)` 最適化）** を Python スクリプトで数分で求め、全4ノードの `/cache/dspark-trace/MARKOV_SCALE` と `config/cluster.json` に反映して `benchmark-matrix` を計測する。
2. **Day 2（小幅カーネル/ロジック修正）**:
   - **案4（`tilelang_copy.py` の累積期待収量修正＋Prior 適正化）** と **案5（`draft_top2_margins` の Confidence Head 統合＋位置別オンライン校正）** を適用し、偽陽性コピーと無駄な検証行を排除して `acceptance_rate` を `0.70+` に乗せる。
3. **Day 3（重み再適合＋検証予算の最終調整）**:
   - **案2（4ランク同期での TP4 `base_logits` top-32 採取 → `W2` 共有変換 $M$ ＋ `W1` 埋め込み正則化 ＋ 1-nat Margin Ranking Loss での共同適合）** で生成した `/cache/dspark-trace/markov_head.pt` を全4ノードに配布し、最後に **案6（`adaptive_verification_row_cost = 0.020〜0.025`）** で増えた採択確率を受理長（`3.45〜3.75 tok/step`）へ変換する。
