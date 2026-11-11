# TP4構成（DGX Spark 4台）のvLLMにおけるDSparkドラフター「Markovバイアス適用前 logits」採取手法の調査レポート

対象ビルドツリー（`goro` 上の最新・654コミット適用済み）:
`/home/masumi/NewModels/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-21d0abdd/src/vllm`

---

## 0. 結論と既存5試行が失敗した真因（コード・行番号による特定）

**結論: TP4構成でドラフターの「Markovバイアス適用前 logits（`base_logits`）」を採取する方法は確実に存在する。**
「TP4のシャード構成では全語彙ドラフトlogitsを実体化する経路が壊れている」という推測は半分正しく、半分は**トレース機構側のランク非対称性・CUDA Graph捕捉・ノードローカル `/cache` マウント**に起因している。そもそも本番のTP4語彙並列経路でも、各ランクはシャード局所のMarkovバイアス適用前 `base_logits`（形状 `[num_reqs, 5, 32320]`, FP32）を `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:574` で**毎ステップ既に計算しており**、さらにその直前の隠れ状態 `sample_hidden`（形状 `[num_reqs * 5, 5120]`, BF16）は**全4ランクで完全に複製（レプリケート）されている**（`vllm/models/deepseek_v4_1/nvidia/dspark.py:494-532`, `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:548-552`）。

既存の5試行（パッチ0050〜0054）がすべて失敗した原因は以下の4つの構造的バグの連鎖である。

### 試行1（Commit `352598216` / パッチ0050）: `_sample_sequential` 内で `vocab_parallel = False` に切替
* **現象**: 起動時に `KernelNotPreparedError: 'finish_argmax' was not prepared: the kernel cache is frozen`。
* **真因**:
  1. DeepSeek-V4.1 の TileLang ランタイムは `vllm/v1/worker/gpu_worker.py:1480-1506`（`_prepare_kernels()`）にて、メモリプロファイル（`determine_available_memory`）の前に全宣言カーネルをコンパイルし、`prepare.freeze(...)`（`gpu_worker.py:1496`）および `activate_jit_monitor(...)`（`gpu_worker.py:1503`）で **TileLang と Triton の JIT コンパイルキャッシュを完全凍結**する。
  2. ドラフターのウォームアップ宣言 `DSparkSpeculator.register_warmups()`（`vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:435-508`）の448行目 `if not self._vocab_parallel_greedy(local_head):` では、起動時に `_vocab_parallel_greedy()` が `True` を返すため、`register_greedy_markov_shard_pair_warmup`（`speculator.py:495` → `finish_argmax_pair_kernel`, `vllm/v1/worker/gpu/sample/tilelang_sampling.py:3737-3744`）のみが登録され、密経路用の `register_sample_greedy_markov_warmup`（`speculator.py:457` → `finish_argmax_kernel`, `tilelang_sampling.py:3747-3750`）が**未登録のままスキップ**された。
  3. 実行時に `vocab_parallel = False` に倒れた瞬間、`sample_greedy_markov()`（`speculator.py:692` → `vllm/v1/worker/gpu/spec_decode/dspark/greedy.py:163` → `tilelang_sampling.py:1007`）が未準備の `finish_argmax_kernel` を呼んで即死した。

### 試行2（Commit `aacd59760` / パッチ0051）: `_vocab_parallel_greedy()` 内で `not dspark_trace.DRAFT_LOGITS_ENABLED` を返す
* **現象**: 起動時 `determine_available_memory` で rank 0 が停止し、他 rank が `RoCE collective ... timeout 10 s`。
* **真因（2つの致命的バグの複合）**:
  1. **モデルの重み分割と `tilelang_markov_head` フラグの不整合（最重要）**:
     `_vocab_parallel_greedy()` は `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:416-433` だけでなく、**モデル構築側 `vllm/models/deepseek_v4_1/nvidia/dspark.py:62-68` にも別関数として定義されている**。パッチ0051は `dspark/speculator.py:422` しか書き換えなかったため、モデル構築側 `DSparkDeepseekV4Model.__init__`（`nvidia/dspark.py:358-372`）では `vocab_parallel=True` のまま `DSparkMarkovHead` が生成され、`markov_w2` は **4ランクにTP分割（各ランク `[32320, 256]`）** された状態で `self.markov_head.markov_w2.tilelang_markov_head = True`（`nvidia/dspark.py:372`）が設定された。
     この状態で `dspark/speculator.py:576` が密経路に入ると:
     * `base_logits = self.model.compute_draft_logits(sample_hidden)`（`speculator.py:576`）は `self.lm_head`（`tilelang_markov_head` 無し）を使うため、`LogitsProcessor._gather_logits`（`vllm/model_executor/layers/logits_processor.py:385-387`）で all-gather され **全語彙 `[16, 5, 129280]`** になる。
     * 一方、ループ内の `bias = self.model.markov_bias(markov_bias_embed)`（`speculator.py:684` → `vllm/model_executor/models/qwen3_dspark.py:111-113`）は `markov_w2` を `LogitsProcessor` に通すが、`_gather_logits`（`logits_processor.py:386`）には以下の条件がある:
       ```python
       if lm_head.tp_size > 1 and not getattr(lm_head, "tilelang_markov_head", False):
           logits = tensor_model_parallel_all_gather(logits)
       ```
       `markov_w2.tilelang_markov_head` が `True` のため **`bias` は all-gather されずシャードサイズ `[16, 32320]` のまま返る**。
     * 直後の `sample_greedy_markov(base_logits[:, i], bias, ...)`（`speculator.py:692` → `tilelang_sampling.py:994-1006`）が、`logits` 幅 `129,280` に対して `bias` 幅 `32,320` のテンソルをカーネル内で `col = 0..129279` まで読み出し、**GPUメモリの境界外読み出し（Out-of-Bounds access）** を起こす。
  2. **`/cache` が4ノード間で共有されておらず各ノードのローカルディスクである点**:
     `/home/masumi/NewModels/spark-ds41f/config/cluster.json:60-61` で `"{checkout}/cache"` が `"/cache"` にバインドマウントされており、NFS共有ではない。`goro`（rank 0）の `/cache/dspark-trace/` だけにマーカーを置くと、`trace.py:64-66, 109` の import 時評価で **rank 0 だけ `DRAFT_LOGITS_ENABLED=True`（密経路・all-gather 実行）、rank 1〜3 は `False`（シャード経路）** に分岐し、`determine_available_memory` の最初の `_dummy_run` で集合通信が不一致となり 10 秒でタイムアウトする。

### 試行3（Commit `f272ec237` / パッチ0052）: `trace.py` で `SPARKNET_ROCE_TIMEOUT_MS=600000` を設定
* **現象**: 効かず（10秒でタイムアウト）。
* **真因**:
  `config/cluster.json:111` がコンテナ起動時の環境変数として `"SPARKNET_ROCE_TIMEOUT_MS": "10000"` を既に注入しているため、`vllm/v1/worker/gpu/spec_decode/dspark/trace.py:74` の `os.environ.setdefault("SPARKNET_ROCE_TIMEOUT_MS", "600000")` は**既存の `"10000"` を上書きできず完全に no-op** であった。また仮に上書きしても、原因は通信遅延ではなくランク間の集合通信不一致およびカーネルクラッシュである。

### 試行4（Commit `406942427` / パッチ0053）: `HIDDEN=0` で入力量ダンプ停止
* **現象**: 症状変わらず。
* **真因**: `_TRACE_HIDDEN`（`trace.py:100, 234`）は `DSparkTrace.record()` 内のホストコピーを止めるだけであり、`_sample_sequential` 内のクラッシュ・集合通信不一致とは無関係。

### 試行5（Commit `40a55387b` / パッチ0054）: 本番経路の横（beside）で `compute_draft_logits(sample_hidden)` を呼ぶ
* **現象**: `CUDA_ERROR_LAUNCH_FAILED`（rank 0 停止＋他 rank のコンテキスト破壊）。
* **真因（2つの致命的バグ）**:
  1. **`self.draft_trace_capture` が rank 0 でしか `True` にならず、rank 0 単独で TP 集合通信（`all_gather`）を発火させた**:
     * `vllm/v1/worker/gpu/spec_decode/dspark/trace.py:397-399` の `maybe_create_dspark_trace()` は `if _TRACE_PATH is None or get_tp_group().rank_in_group != 0: return None` となっており、**`self._dspark_trace` は rank 0 にしか存在しない（rank 1〜3 では `None`）**。
     * そのため `vllm/v1/worker/gpu/model_runner.py:3483-3486`（`_capture_dspark_trace_logits`）の `if trace is None: return` により、**`self.speculator.draft_trace_capture` は rank 0 でのみ `True` になり、rank 1〜3 では永久に `False`** となる。
     * その結果、`dspark/speculator.py:562-566, 579-585` の `if want_draft_logits:` に **rank 0 だけが入り**、`cap = self.model.compute_draft_logits(sample_hidden)`（`speculator.py:585`）の中で `LogitsProcessor._gather_logits`（`logits_processor.py:386`）の **`tensor_model_parallel_all_gather(logits)` を rank 0 単独で呼び出した**。
     * 一方 **rank 1〜3 は `if want_draft_logits:` をスキップ**して `speculator.py:662` の `tensor_model_parallel_all_gather(pairs, dim=-1)`（256バイトの Sparknet RoCE one-shot 通信）に入った。
     * `cuda_communicator.py:486-496` の docstring に明記されている通り、Sparknet の one-shot RoCE 集合通信はタイムアウトすると **GPU を強制停止させ、次の CUDA 同期で `CUDA_ERROR_LAUNCH_FAILED` を発生させる**。これが観測された `CUDA_ERROR_LAUNCH_FAILED` の正体である。
  2. **`DFlashCudaGraphManager` の FULL CUDA Graph 捕捉により、通常の decode ステップでは `_sample_sequential` の Python 分岐自体が実行されない**:
     * `DFlashSpeculator` は `CUDAGraphMode.FULL_DECODE_ONLY`（`vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:158`）を使用し、起動時の `capture_model`（`dflash/speculator.py:503`）で `self.draft_trace_capture = False` の状態のまま `_generate_draft` → `_sample_sequential` 全体を FULL CUDA Graph に捕捉する。
     * 実行時の decode ステップでは `dflash/speculator.py:905-920` で `self.query_cudagraph_manager.run_fullgraph(batch_desc)` がリプレイされるため、**`need_eager=True` に落とすか、CUDA Graph 内に組み込んだ固定バッファへ書き出さない限り、`_sample_sequential` 内の Python `if want_draft_logits:` 分岐はそもそも実行されない**（グラフ未捕捉のバッチ形状や prefill 直後で eager に入った瞬間に上記1のランク単独 `all_gather` が発火してクラッシュした）。

---

## 1. 既存機構で使えるものは無いか（コード調査結果）

### 1.1 `self.draft_logits` が非 `None` になる条件・`draft_sample_method`・`rejection_sample_method` との関係
* **`self.draft_logits` の確保条件**:
  * `DraftModelSpeculator.__init__`（`vllm/v1/worker/gpu/spec_decode/speculator.py:184-198`）にて、`self.speculative_config.draft_sample_method == "probabilistic"` のときのみ `[max_num_reqs, num_speculative_steps, vocab_size]` が確保される。`draft_sample_method == "greedy"` のときは `self.draft_logits = None`（`speculator.py:184`）。
  * さらに `DFlashSpeculator.load_model`（`vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:263-283`）にて、`draft_sample_method == "probabilistic"` かつ `tp_size > 1` かつドラフターが vocab-parallel 対応の場合、`self.draft_logits` は **シャード幅 `[max_num_reqs, num_speculative_steps, local_vocab(32320)]` に縮小** され、`VPDraftCache`（`vllm/v1/worker/gpu/spec_decode/vocab_parallel.py:36-57`）に登録される。
* **Markovバイアス適用前 logits 採取に使えるか → 使えない（不適）**:
  * `DSparkSpeculator._sample_sequential`（`vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:700-703`）および ` _sample_logits`（`speculator.py:510-529`）が `self.draft_logits` に書き込むのは `logits_i = base_logits[:, i] + bias`、すなわち **Markovバイアス適用後（post-bias）の logits** であり、適用前（pre-bias）の `base_logits` ではない。
  * さらに `draft_sample_method="probabilistic"` に変更すると `_vocab_parallel_greedy()` が `False`（`dspark/speculator.py:422`, `nvidia/dspark.py:65`）となり、ドラフターの挙動自体が Gumbel サンプリングに変わってしまう上、`nvidia/dspark.py:372` の `tilelang_markov_head = True` は残るため、様々な副作用が生じる。

### 1.2 `acceptance_estimator` 経由の `_needs_logits_for_acceptance()` が `True` になる構成
* **条件**:
  * `DSparkSpeculator._needs_logits_for_acceptance()`（`vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:130-134`）は `not self.use_confidence_head and getattr(self, "acceptance_estimator", None) is not None` のとき `True` を返す（すなわち `confidence_head` を無効化し、`enable_adaptive_verification=true` のままにした場合）。
* **そのときの TP>1 の挙動 → 全語彙 logits は実体化されず、そもそもバイアス適用後である**:
  * `dspark/speculator.py:416-433` の `_vocab_parallel_greedy()` は `self.use_confidence_head` をチェックしないため、TP4 & greedy では依然として `vocab_parallel = True` のままとなり、`speculator.py:631-683` のシャード経路を通って `continue` するため、722行目の `acceptance_estimator.predict(...)` にすら到達しない。
  * 仮に `vocab_parallel = False` 側（`speculator.py:700, 735`）に入ったとしても、`acceptance_estimator.predict` に渡されるのは `logits_i = base_logits[:, i] + bias`（Markovバイアス適用後）であり、`OnlineAcceptanceEstimator.predict`（`vllm/v1/worker/gpu/spec_decode/acceptance_estimator.py:554-606`）はスカラー特徴量 `logit(max q)`（`self.features`）と採択確率 `self.predictions` しか保存しない。

### 1.3 他実装（`multi_module_mtp` / `eagle`）や `LogitsProcessor` にある既存機構
* **`multi_module_mtp` / `eagle`**:
  * `vllm/v1/worker/gpu/spec_decode/multi_module_mtp/speculator.py` および `vllm/v1/worker/gpu/spec_decode/eagle/` を全検索したが、logits をホストへダンプする既存のデバッグ経路は**存在しない**（`eagle/utils.py:56` の `.cpu()` はロード時の重み一致確認のみ）。
* **`LogitsProcessor.get_top_k_tokens`（使える既存機構！）**:
  * `vllm/model_executor/layers/logits_processor.py:574-633` に **`LogitsProcessor.get_top_k_tokens(lm_head, hidden_states, k)`** が既に実装されている。
  * このメソッドは TP>1 において全語彙を all-gather せず、各ランクで `self._get_logits(lm_head, hidden_states, skip_gather=True)` によりシャード局所 logits `[N, 32320]` を計算し、各シャードで `torch.topk(logits, k, dim=-1)` を取って `[top_k_logits, top_k_indices.to(logits.dtype)]`（形状 `[N, 2*k]`）だけを `tensor_model_parallel_all_gather` し、グローバル top-k を返す。
  * `k=64, N=80`（`16 reqs * 5 steps`）でも通信量は各ランク `80 * 128 * 4 = 40,960 bytes (40 KB)` であり、`SPARKNET_ROCE_ALLGATHER_MAX_BYTES=2097152`（2 MB, `config/cluster.json:285`）より2桁小さく、未準備の TileLang カーネルも一切呼ばない。

### 1.4 `--enforce-eager` や `cudagraph_mode` 変更で何が救われるか
* **カーネルキャッシュ凍結（`KernelNotPreparedError`）は救われない**:
  * `Worker._prepare_kernels()`（`vllm/v1/worker/gpu_worker.py:1455-1506`）は CUDA Graph の有無（`--enforce-eager` や `cudagraph_mode`）とは**完全に独立して**モデルロード直後に実行され、`prepare.freeze(...)`（`gpu_worker.py:1496`）でキャッシュを凍結する。
* **救われるもの**:
  * `DFlashSpeculator` の FULL CUDA Graph リプレイ（`vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:905-920`）がバイパスされ、毎ステップ Python の `_generate_draft` → `_sample_sequential` が実行されるようになる。
  * ただしクラスタ全体で `--enforce-eager` にしなくても、`DFlashSpeculator.propose` の `need_eager`（`dflash/speculator.py:888`）を**採取ステップのみ全4ランク同期で `True` にする**（後述の設計案2）、あるいは **CUDA Graph 内で固定バッファへ `copy_` する**（設計案1・設計案3）ことで、通常ステップの FULL CUDA Graph 性能を維持したまま採取できる。

---

## 2. 成立する設計案（実装コストとリスクの低い順・3案）

前提として、`/cache` は各ノードのローカルディレクトリ（`config/cluster.json:60-61`）であるため、**どの案を採る場合も `/cache/dspark-trace/` 配下のマーカーファイル（`ENABLED`, `DRAFTTOPK`, `EVERY` 等）は必ず4ノード全て（`goro`, `shiro`, `taro`, `jiro`）に同一内容で配置すること**。

---

### 設計案1（最推奨・最小リスク）: シャード局所 `base_logits` を再利用し、4ランク同期 `need_eager` でシャード別 `topk` + 小サイズ `all_gather` を行う
本番のTP4経路が `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:574` で**既に計算しているシャード局所の `base_logits`（`[num_reqs, n_spec, 32320]`, FP32、Markovバイアス適用前）** をそのまま使う。新たな `lm_head` GEMM も、未準備の TileLang カーネルも、41 MB の全語彙 `all_gather` も一切使わない。

#### 変更するファイル:行と最小コード差分スケッチ
1. **`vllm/v1/worker/gpu/model_runner.py:3475-3495`（`_capture_dspark_trace_logits`）**:
   `self.speculator.draft_trace_capture` が rank 0 だけでなく **全4ランクで同一ステップに `True` になる** よう、ランナー側に全ランク共通のステップカウンタ `_dspark_trace_step` を持たせる。
   ```python
   # vllm/v1/worker/gpu/model_runner.py:3480-3490 付近
   def _capture_dspark_trace_logits(self, logits: torch.Tensor) -> None:
       from vllm.v1.worker.gpu.spec_decode.dspark import trace as dspark_trace
       self._dspark_logits_topk = None
       if self.speculator is not None:
           self.speculator.draft_trace_capture = False
       if not dspark_trace.ENABLED:
           return
       self._dspark_trace_step = getattr(self, "_dspark_trace_step", 0) + 1
       if self._dspark_trace_step % dspark_trace._TRACE_EVERY != 0:
           return
       if self.speculator is not None and dspark_trace.DRAFT_LOGITS_ENABLED:
           # 全TPランクで同時に True にする（集合通信の同期を保つため必須）
           self.speculator.draft_trace_capture = True
       if self._dspark_trace is not None:
           self._dspark_logits_topk = self._dspark_trace.capture_logits(logits)
   ```
   （※ `trace.py:213` の `DSparkTrace.record()` 側の `self._step` インクリメントと二重判定にならないよう、`trace.py:66` に `ENABLED = _TRACE_PATH is not None` を公開する）

2. **`vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:881-890`（`DFlashSpeculator.propose`）**:
   採取ステップ（`self.draft_trace_capture == True`）のみ、全4ランク同時に `need_eager=True` として FULL CUDA Graph をバイパスし Python の `_sample_sequential` に入る。
   ```python
   # vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:881-890
   batch_desc, batch_sync = dispatch_cg_and_sync_dp(
       self.query_cudagraph_manager,
       num_reqs,
       num_batch_tokens,
       uniform_token_count=self.num_query_per_req,
       dp_size=self.dp_size,
       dp_rank=self.dp_rank,
       need_eager=is_profile or getattr(self, "draft_trace_capture", False),
       dp_sync=batch_sync,
   )
   ```

3. **`vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:579-589`（`DSparkSpeculator._sample_sequential`）**:
   `self.model.compute_draft_logits(sample_hidden)` を呼ばず、直前の574行目で計算済みのシャード局所 `base_logits`（`[num_reqs, n_spec, 32320]`）に対して各ランクで `torch.topk` を取り、`[num_reqs * n_spec, 2 * k]`（40 KB）だけを `tensor_model_parallel_all_gather` する（`LogitsProcessor.get_top_k_tokens` `logits_processor.py:604-633` と同一アルゴリズム）。
   ```python
   # vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:579-589
   if want_draft_logits:
       k = min(dspark_trace._TRACE_DRAFT_TOPK, shard_vocab if can_unfuse_markov else vocab_size)
       if vocab_parallel:
           v_start, s_vocab = self.model.draft_vocab_shard
           local_slice = base_logits[:, :, :s_vocab].reshape(num_sample, s_vocab)
           top_v, top_i = torch.topk(local_slice, k, dim=-1)
           top_i = top_i + v_start
           packed = torch.cat([top_v.float(), top_i.to(torch.float32)], dim=-1)
           gathered = tensor_model_parallel_all_gather(packed, dim=-1)  # [num_sample, tp * 2 * k]
           if get_tensor_model_parallel_rank() == 0:
               tp = get_tensor_model_parallel_world_size()
               g = gathered.view(num_sample, tp, 2, k)
               all_v = g[:, :, 0, :].reshape(num_sample, tp * k)
               all_i = g[:, :, 1, :].to(torch.int32).reshape(num_sample, tp * k)
               gv, gi = torch.topk(all_v, k, dim=-1)
               gi_tok = torch.gather(all_i, -1, gi)
               self._dspark_draft_topk = (gv.cpu(), gi_tok.cpu())
       else:
           self._dspark_draft_topk = dspark_trace.capture_draft_logits(
               base_logits, num_reqs, n_spec
           )
   ```

#### なぜ今度は落ちないか
* **カーネル経路**: `base_logits` は本番経路の `compute_draft_logits_local`（準備済み）の出力そのものであり、`torch.topk` / `torch.gather` は PyTorch ATen 標準カーネル（TileLang / Triton JIT 監視対象外）であるため、`KernelNotPreparedError` は発生しない。
* **集合通信**: 全4ランクが同一ステップで `draft_trace_capture = True` → `need_eager = True` となり、全4ランクが `tensor_model_parallel_all_gather(packed, dim=-1)` を同時に呼ぶ。テンソルサイズは1ランクあたり最大 `80 * 128 * 4 = 40 KB`（16バイト境界整列）であり、`SPARKNET_ROCE_ALLGATHER_MAX_BYTES=2 MB` 以下なので Sparknet one-shot RoCE（または準備済みの `rank_rows_kernel`, `vllm/models/deepseek_v4_1/nvidia/model.py:790`）でマイクロ秒オーダーで完了する。
* **メモリ**: 追加の永続バッファは 0 バイト、一時テンソルも数十 KB のみ。`determine_available_memory` 中は `_capture_dspark_trace_logits` が呼ばれないため `draft_trace_capture = False` のままであり、起動時のメモリプロファイル・CUDA Graph 捕捉に一切影響しない。

#### 事前に安く検証する方法
1. 4ノード全てで `/cache/dspark-trace/{ENABLED,DRAFTTOPK,EVERY}` の存在と内容が一致していることを `for h in goro shiro taro jiro; do ssh $h head /cache/dspark-trace/*; done` で確認する。
2. クラスタを起動する前に `python3 -m py_compile` で変更3ファイルの構文と、`python3 -c` で `packed -> view(num_sample, tp, 2, k)` の復元ロジックを単体検証する（`129,280 < 2^24` のため `int32 -> float32 -> int32` 変換はビット完全一致）。

#### 採取できる情報の粒度
* **Markovバイアス適用前（pre-bias）の厳密なグローバル Top-$K$（例: $K=64$）の logit 値（FP32）とトークン ID（INT32）**。既存の `.drafttopk.f32` / `.drafttopk.i32` フォーマットと完全互換。
* （もし全語彙 129,280 個が必要な場合は、`num_reqs=1` 等のリプレイ実行時に `tensor_model_parallel_all_gather(base_logits.to(torch.bfloat16), dim=-1)` を4ランク同期で呼べば、`[5, 129280]` BF16 = 1.29 MB < 2 MB となり全語彙も同様に採取可能。）

---

### 設計案2（集合通信ゼロ・CUDA Graph 完全維持・全語彙復元可能）: Rank 0 で既にレプリケートされているドラフター出力隠れ状態 `sample_hidden` を採取し、オフラインで `lm_head(norm(sample_hidden))` を計算する
コードを精査すると、そもそもドラフターの3層 MTP トランスフォーマ（`mtp.0`〜`mtp.2`）の出力 `head_hidden`（`vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:798`）およびそこから抽出した `sample_hidden`（`[num_reqs * 5, 5120]`, BF16, `dspark/speculator.py:548-552`）は、**`mhc.post_collapse`（`vllm/models/deepseek_v4_1/nvidia/dspark.py:523`）の時点で全4ランクに完全レプリケート（同一値）されている**。
語彙が4ランクに分割されるのはその直後の `self.lm_head`（`nvidia/dspark.py:696-700`）だけである。
（※なお現在 `model_runner.py:3793` が `.hidden.f16` にダンプしている `self.speculator.hidden_states` は、`dflash/speculator.py:682` の `context_states` すなわちドラフター**入力前**の `main_x` であり、ドラフター**出力後**の `sample_hidden` ではない。）

したがって、**Rank 0 だけで `sample_hidden`（`[num_reqs * 5, 5120]`, BF16 = 最大 800 KB）を固定バッファに `copy_` してホストへ書き出せば、TP 集合通信を1バイトも発生させずに、オフラインでチェックポイントの `mtp.2.norm.weight`（`[5120]`）と `lm_head.weight`（`[129280, 5120]`）を掛けて 129,280 全語彙の厳密な Markov バイアス適用前 `base_logits` を計算できる**。さらに `confidence_head`（入力次元 `5120 + 256 = 5376`, `vllm/model_executor/models/qwen3_dspark.py:217`）の学習に必要な `sample_hidden` そのものも同時に手に入る。

#### 変更するファイル:行と最小コード差分スケッチ
1. **`vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:249` 付近（`DSparkSpeculator.__init__`）**:
   固定バッファ `self._trace_sample_hidden`（`[max_num_reqs * n_spec, hidden_size]`, BF16 = 800 KB）を確保する。
   ```python
   # vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:255 付近
   self._trace_sample_hidden: torch.Tensor | None = None
   if dspark_trace.DRAFT_LOGITS_ENABLED and get_tensor_model_parallel_rank() == 0:
       self._trace_sample_hidden = torch.empty(
           (self.max_num_reqs * self.num_speculative_steps, self.model.config.hidden_size),
           dtype=self.dtype,
           device=self.device,
       )
   ```
2. **`vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:552` 直後（`_sample_sequential`）**:
   CUDA Graph 捕捉内にもそのまま入る無条件の `copy_`（Device-to-Device 800 KB コピー、所要 `< 2 μs`）を1行入れる。
   ```python
   # vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:553
   if self._trace_sample_hidden is not None:
       self._trace_sample_hidden[:num_sample].copy_(sample_hidden)
   ```
3. **`vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:735-777`（`propose` の末尾、CUDA Graph リプレイ完了後）**:
   `self.draft_trace_capture` が `True` のステップで、Rank 0 が `self._trace_sample_hidden[:num_reqs * n_spec].detach().cpu()` を取り出す（または Rank 0 のローカルシャード分だけでなく、ホストへダンプしてオフラインで `lm_head` を掛ける）。

#### なぜ今度は落ちないか
* **カーネル経路**: `Tensor.copy_` は CUDA ランタイムの `cudaMemcpyAsync(DeviceToDevice)` であり、TileLang / Triton JIT カーネルを一切使わない。
* **集合通信**: **追加の集合通信が完全にゼロ（0回）**。Rank 0 のローカルメモリコピーだけで完結するため、RoCE タイムアウトやランク間デッドロックが原理的に起こり得ない。
* **CUDA Graph**: `self._trace_sample_hidden[:num_sample].copy_(sample_hidden)` は固定アドレス間のコピーとして `DFlashCudaGraphManager` の FULL CUDA Graph 内にそのまま記録され、通常の FULL Graph リプレイ時にも自動で最新ステップの `sample_hidden` が書き込まれる。
* **メモリ**: 固定バッファは `16 * 5 * 5120 * 2 = 819,200 bytes`（**0.78 MiB**）のみ。

#### 事前に安く検証する方法
* オフラインの単体スクリプト（CPU または単一GPU）で、チェックポイントの `mtp.2.norm.weight` と `lm_head.weight` のうち rank 0 のシャード（先頭 32,320 行）を取り出し、`RMSNorm(sample_hidden) @ lm_head_shard.T` が `compute_draft_logits_local(sample_hidden)` と一致することを1ステップ分のダミー入力で確認できる。

#### 採取できる情報の粒度
* **129,280 全語彙すべての厳密な Markov バイアス適用前 `base_logits`**（オフライン行列積による完全復元）＋ **`confidence_head` への入力隠れ状態 `sample_hidden`（5,120次元 BF16）そのもの**。

---

### 設計案3: CUDA Graph 内でシャード局所 `base_logits` を固定バッファへ `copy_` し、`propose()` 終了後（Graph 外）に全4ランクで Top-$K$ 還元する
「エンジン内で直接 `.drafttopk.f32/.i32` を出力したいが、採取ステップでも `DFlashCudaGraphManager` の FULL CUDA Graph を `need_eager=True` に落としたくない」場合の設計案。

#### 変更するファイル:行と最小コード差分スケッチ
1. **`vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:345` 付近（`load_model` 内）**:
   全4ランクで固定バッファ `self._trace_local_base_logits`（形状 `[max_num_reqs, n_spec, 32320]`, FP32 = **9.86 MiB**）を確保する。
2. **`vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:579`（`_sample_sequential` 内）**:
   ```python
   if self._trace_local_base_logits is not None:
       self._trace_local_base_logits[:num_reqs].copy_(base_logits)
   ```
   この `copy_` は無条件実行のため `DFlashCudaGraphManager` の FULL CUDA Graph にそのまま捕捉される。
3. **`vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:920` の直後（`propose` の末尾、CUDA Graph 外）**:
   設計案1と同様に全4ランク共通のステップ判定（`self.draft_trace_capture == True`）の下で、CUDA Graph の外で `self._trace_local_base_logits[:num_reqs]` からシャード別 `torch.topk(k)` ＋ `tensor_model_parallel_all_gather(packed, dim=-1)`（40 KB）を実行し、Rank 0 で `self._dspark_draft_topk` に格納する。

#### なぜ今度は落ちないか
* `_sample_sequential`（CUDA Graph 内）には `copy_` しか追加しないため、CUDA Graph の分岐制約・カーネル凍結・グラフ内集合通信に一切触れない。
* 集合通信（40 KB の `all_gather`）は `propose()` の最後（CUDA Graph の外）で全4ランク同期で呼ばれる。

#### 事前に安く検証する方法
* 9.86 MiB の固定バッファ追加が KV キャッシュブロック数を減らさないことを起動ログの `Available KV cache memory` で確認する（GB10 128GB 上で 9.86 MiB は誤差範囲）。

#### 採取できる情報の粒度
* **Markovバイアス適用前（pre-bias）の厳密なグローバル Top-$K$（FP32値・INT32 ID）**。FULL CUDA Graph 実行を維持したまま採取可能。

---

## 3. 代替案（エンジンを大きく変えずに済むもの）の実現性と情報量の限界

### 3.1 各 rank の `(max, idx)` ペアを all-gather してグローバル top-k を作る案
* **コード上の位置**:
  `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:649-663` にて、各ステップ `i` で `markov_greedy_pairs_local`（または `greedy_markov_shard_pair`）が `self._greedy_shard_pairs[:num_reqs]`（形状 `[num_reqs, 4]`）に各シャードの 1位 `(max_val, argmax_idx)`（`top2_branch` 有効時は 2位までの4列）を書き込み、662行目で `tensor_model_parallel_all_gather(pairs, dim=-1)`（形状 `[num_reqs, 16]`）を実行している。
* **実現性**:
  すでに662行目で all-gather されたテンソルが全ランク上に存在するため、それをホストへ保存するだけであれば通信追加ゼロで可能。また、ペアに入っている値は **Markovバイアス適用後**（$\text{base\_logit}[t] + W_2[t] \cdot W_1[\text{prev}]$）だが、チェックポイントの `markov_head.embed.weight`（$W_1$）と `markov_head.head.weight`（$W_2$）および直前トークン `prev` は既知であるため、オフラインで $W_2[t] \cdot W_1[\text{prev}]$ を引き算すれば、そのトークン $t$ の **Markovバイアス適用前 `base_logit[t]` を厳密に逆算できる**。
  さらに、`/cache/dspark-trace/MARKOV_SCALE` に `0.0` を書いて起動すれば（`speculator.py:92-104` の `_load_markov_scale()`、パッチ0048）、実行時の Markov バイアス自体が $0$ になるため、各シャードの純粋な `base_logits` の top-1（`top2_branch` 時は top-2）がそのまま得られる。
* **情報量の限界**:
  * 1ステップあたり最大でも **4トークン分**（各32,320語彙シャードの1位×4ランク、`top2_branch` でも最大8トークン分）しか得られない。
  * 同一シャード内に上位トークンが複数集中している場合（例: rank 0 のシャードにグローバル1位〜10位が全て入っている場合）、2位以下（または3位以下）の情報は完全に失われるため、残差フィッティング（Markov head 学習）には情報量が不足する。

### 3.2 `select_first_argmax` の出力を流用する案
* **コード上の位置**:
  `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:661-681` および `vllm/v1/worker/gpu/spec_decode/dspark/greedy.py:166-214`（実装は `vllm/v1/worker/gpu/sample/tilelang_sampling.py:1132-1193`）。
* **実現性と情報量の限界**:
  `select_first_argmax` カーネルが出力するのは、グローバル1位のトークンID `out`（`self.draft_tokens[:num_reqs, i]`, `int64`）と、`top2_branch` 有効時の2位トークンID `self.draft_runner_up_tokens[:num_reqs, i]`（`int64`）および top1-top2 マージン `self.draft_top2_margins[:num_reqs, i]`（`float32`, ただしバイアス適用後）のみであり、**logit 値そのものは出力テンソルに書き出さずに破棄している**（`tilelang_sampling.py:1175-1190`）。したがって `select_first_argmax` の出力だけでは pre-bias logits の値は一切得られない。

### 3.3 `confidence_head` の入力（`main_x` / `.hidden.f16`）や出力確率から逆算する案
* **コード上の位置**:
  * `.hidden.f16` にダンプされているテンソル: `vllm/v1/worker/gpu/model_runner.py:3793` の `self.speculator.hidden_states`（＝ `dflash/speculator.py:682, 710` で作られる `context_states` / `main_x`、すなわちターゲットの補助隠れ状態を結合した **ドラフター入力**）。
  * `confidence_head` の計算: `vllm/models/deepseek_v4_1/nvidia/dspark.py:836-845` および `vllm/model_executor/models/qwen3_dspark.py:215-219`（ドラフター3層通過後の `sample_hidden` `[5120]` と `markov_embed` `[256]` を結合した 5,376 次元ベクトルに `confidence_head.proj.weight` `[1, 5376]` を掛けてスカラー確率 1 個を出力）。
* **実現性と情報量の限界**:
  1. **`confidence_head` の出力（スカラー1個）からの逆算**: $5376 \to 1$ の線形射影はランク1であり、5,120次元の `sample_hidden` や 129,280次元の `base_logits` を逆算することは数学的に不可能。
  2. **既存の `.hidden.f16`（`main_x`）からのオフライン順伝播による再計算**:
     `main_x` はドラフターの3層の MTP トランスフォーマ（`mtp.0`, `mtp.1`, `mtp.2`、`nvidia/dspark.py:494-533`）へのコンテキスト KV 入力である。ここから `sample_hidden` と `base_logits` を得るには、オフラインで **系列全体のコンテキストに対する3層分の MLA Sliding-Window Attention + MoE（ルーティング・エキスパート計算）+ mHC を完全再現**しなければならず、事実上ドラフターモデル全体を別プロセスで推論し直すのと同じコストがかかる（さらに `use_context_graph=True` のステップでは `dflash/speculator.py:683, 827` により `self._context_preparer.run` が直接 KV キャッシュへ書き込むため、`self.speculator.hidden_states` 自体が更新されないケースがある）。
     したがって、`main_x` から再計算するよりも、**設計案2の通り3層通過後の `sample_hidden`（`dspark/speculator.py:552`）をダンプする方が圧倒的に簡単かつ厳密**である。

---

## 4. やらない方がよい案とその理由

1. **`_vocab_parallel_greedy()` を `dspark/speculator.py` だけ `False` に書き換えて密経路（`vocab_parallel=False`）へ落とす案（試行1・試行2の延長）**
   * **理由**:
     * 前述の通り、`vllm/models/deepseek_v4_1/nvidia/dspark.py:62-68` にも同名の `_vocab_parallel_greedy()` が存在し、かつ `nvidia/dspark.py:372` で `self.markov_head.markov_w2.tilelang_markov_head = True` がハードコードされている。
     * そのため `LogitsProcessor._gather_logits`（`vllm/model_executor/layers/logits_processor.py:386`）が `markov_w2` の all-gather をスキップし、`base_logits`（幅 129,280）と `bias`（幅 32,320）の形状不一致で GPU メモリ境界外アクセスを引き起こす。
     * 仮に `nvidia/dspark.py:62-68` と `372` 行目まで全て書き換えて密経路を通したとしても、毎ステップ `[80, 129280]` FP32（**41.37 MB**）の `all_gather` が走り、`SPARKNET_ROCE_ALLGATHER_MAX_BYTES=2 MB`（`config/cluster.json:285`）を大幅に超えるため one-shot RoCE から PyNCCL フォールバックに切り替わり、さらに `markov_w2`（66 MB）が全ランクに重複保持される。

2. **Rank 0 だけで有効になるフラグ（`self._dspark_trace is not None` 等）の分岐内で `compute_draft_logits` や `all_gather` を呼ぶ案（試行5の延長）**
   * **理由**:
     * `trace.py:398` の `maybe_create_dspark_trace()` は Rank 0 でしかオブジェクトを返さない。Rank 0 だけが `compute_draft_logits`（内部で `tensor_model_parallel_all_gather` を呼ぶ、`logits_processor.py:386`）に入ると、Rank 1〜3 が別の集合通信（`speculator.py:662` の `pairs` all-gather）で待ち続け、Sparknet RoCE ランタイムが 10 秒後に GPU を停止させて `CUDA_ERROR_LAUNCH_FAILED`（`cuda_communicator.py:486-496`）となる。
     * TP 集合通信を含む処理を入れる場合は、**必ず全4ランクで同一の値になる条件式**で分岐させなければならない。

3. **`DFlashCudaGraphManager` の FULL CUDA Graph を意識せずに `_sample_sequential` 内に Python の動的 `if` 分岐だけを書く案**
   * **理由**:
     * `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:905-920` の通り、通常の decode ステップは起動時にキャプチャされた FULL CUDA Graph（`run_fullgraph`）をリプレイするだけなので、キャプチャ時に `False` だった Python の `if` 分岐には実行時に一切入らない。
     * 採取ステップで `need_eager=True`（`dflash/speculator.py:888`）を全4ランク同時に立てる（設計案1）か、CUDA Graph 内に常時実行の `Tensor.copy_` を入れて Graph の外で読み出す（設計案2・設計案3）必要がある。

4. **`goro`（ヘッドノード）の `/cache/dspark-trace/` だけを操作する運用 / `trace.py` 内で `os.environ.setdefault("SPARKNET_ROCE_TIMEOUT_MS", ...)` を使う案**
   * **理由**:
     * `config/cluster.json:60-61` の通り `/cache` は各ノードのローカルディレクトリであり、4ノード全てでファイルを同期しないと import 時の定数（`DRAFT_LOGITS_ENABLED` 等）がランク間で食い違う。
     * また `SPARKNET_ROCE_TIMEOUT_MS` は `config/cluster.json:111` で `"10000"` が事前設定されているため `os.environ.setdefault` は無効であり、かつ C++ 通信器の初期化後に Python 側で環境変数を書き換えても反映されない。
