# OPERATIONS SPEC — on-device SP-chat runtime (unified, router-free)

最新運用版（`baya1116/hypernet-sp-distill` の正典 `hypernet_sp/app_session_torch.py` に本リポジトリの
パッチ `faithful_patches/app_session_torch.patch` を当てた状態）の運用仕様書。
DeepSeek-R1-Distill-Qwen-1.5B(FFT済) を、SP圧縮による O(1) 有界メモリ＋階層メモリ＋オンデマンド
検索（BGEリコール / Web）で動かすオンデバイス・チャット。

最終検証は RunPod GPU(RTX A5000) 実機。各ターン 1–5s(生成のみ) / Web ターンは 60–100s
(Wikipedia 取得込み)。

---

## 0. このバージョンの要点（旧仕様からの差分）
**intent ルーター(6分岐: fact-ack / recall / lookup→web / command / math / chitchat)を撤去。**
毎ターンを単一の生成経路にし、メモリと Web を「retrievability ゲート」でオンデマンド注入する方式に一本化。

- 旧: 質問を6クラスに分類して分岐 → 雑談が "fact" 扱いで定型ackされる/既知質問がオフラインで拒否、等の誤動作。
- 新: 分類しない。**BGE 検索で当たれば想起注入、当たらず知識質問なら Web、どちらも無ければ素で生成。**

---

## 1. 構成物（全て公開 HF `baya1116/hypernet-sp-distill` から取得可）
| 物 | パス |
|---|---|
| 学生モデル(FFT) | `fft_out/student.pt` → `build_fft_hf.py` で `fft_hf/` 化 |
| SP pooler | `fft_out/pooler.pt`（`attn_export3_torch.load_pooler`） |
| ランタイム | `hypernet_sp/app_session_torch.py`(正典) + `memory_core/intent_route/anaphora/web_guard/decode_policy/calculator` |
| Web検索 | `runtime/web_search.py`(DuckDuckGo→Wikipedia) ／ `runtime/rag.py`(BGE `BAAI/bge-small-en-v1.5`) |
| 分類器(specificity) | `evals/specificity_clf.joblib`（固有値 pin 判定にのみ使用。intent_clf は不使用） |
| **本バージョンのパッチ** | 本リポジトリ `cotscan_live/faithful_patches/app_session_torch.patch` |

---

## 2. セットアップ / 起動
```
# 再構築（公開HFから）→ パッチ適用
python3 faithful_setup.py          # faithful_run/ を構築（重み・モジュール取得）
cd faithful_run && python3 build_fft_hf.py && cd ..
python3 faithful_fixes.py          # faithful_patches/app_session_torch.patch を適用

# 対話（CPUは遅い。GPUは SPCHAT_DEVICE=cuda）
python3 demo_chat.py               # REPL（--no-web でWeb無効 / 既定はWeb有効）
```
GPU 一括は `cotscan_live/boot_runpod.sh`（公開HF再構築→パッチ→バッテリ→結果を:8000で配信）。

主要ダイヤル: `rw`(raw window, 既定512, 512–1024で品質↑/KV~30MB) / `cap`(生成上限) / Web on-off。

---

## 3. ターン・パイプライン（契約・本バージョン）
入力 `user_msg` → 出力 `(answer, source, chunks)`。`AppSession.turn()`。

```
1. 明示保存コマンド  "remember/save..."＋具体値 → persist+pin して即ack（生成なし）
2. specificity pin   固有値(コード/数値/固有名)があれば WM に pin（上限12）
3. 想起ゲート(recall) retrieve_personal: L1(セッション)+L2(ディスク)+pins を BGE 検索。
                     ※必ず「現ターンをメモリ保存する“前”」に検索（自己マッチ防止）。
                     低確信(<0.62)ヒットは破棄（無関係メモを“closest note”で出さない）。
4. 算術文脈          back-reference("add 10 to that"等)のある compute だけ直近の結果を注入。
5. Webゲート         memory ヒット無し＋疑問文＋Web有効 → DuckDuckGo→Wikipedia を引き injection-guard。
6. 生成             SP-evict ループ(rw=512, maxD=4096) ＋ DecodePolicy(think内temp0.6/以降greedy)。
                     ・MQ前置きに常時 persona（拒否・崩壊の抑止）。
                     ・chunks高確信時のみ「文脈から逐語引用」テンプレ、低確信は通常生成。
7. 後処理           calculator 検算 ／ compute結果を“キー付き宣言文”で保存(検索可) ／
                     表示正規化(\boxed・特殊トークン除去) ／ 拒否文は高温で1回リトライ。
```

---

## 4. 適用済みパッチ（会話で発見→修正、`faithful_patches/app_session_torch.patch`）
| # | 問題 | 修正 |
|---|---|---|
| FIX2 | `\boxed{}` 等 LaTeX 漏れ | 表示正規化で除去 |
| FIX3 | 良性雑談を拒否/崩壊 | MQ前置きに常時 persona |
| FIX4 | ルーターが雑談をfact扱い/既知Qをオフライン拒否 | **6分岐撤去→ゲート式一本化** |
| FIX6 | 開放質問がゆるい想起で"closest note" | 低確信(<0.62)検索を破棄し通常回答 |
| FIX7 | 算術follow-up("add 10 to that")が前回結果無視 | back-ref付き算術をcompute扱いに |
| FIX8 | 会話フレーム実況/role-play・特殊トークン漏れ | フレーム撤回＋拒否は高温retry＋特殊トークン除去 |
| A   | 計算結果が上書きされ recall 不能 | 結果を**キー付き宣言文で保存**(同キーのみ上書き) |

---

## 5. 能力（実機バッテリ観測・正直版）
### できる（安定）
- 拒否しない雑談 / 既知事実の即答（オフライン: capital of France→Paris）
- 固有値の保存と**想起**（flight→JL412, locker→C12, 複数列挙も可。distraction後も維持）
- 単発計算（18×7→126, follow-up「add 10 to that」→136）
- **Web 連携**（src=L3·web）: capital of Australia→Canberra, Pride and Prejudice→Jane Austen,
  symbol for gold→Au（いずれも Wikipedia 出典付き）
- **複合(recall×web)**: 同一会話で Web ルックアップと個人想起を交互にしても**干渉なし**

### 不安定/不可（主に1.5Bの容量＋検索順位）
- **多段の会話内計算**（パーティ文章題の連鎖）: 前半の単発は正、数値が積もると後半が混同
- 世界知識の**細部の正確性**（要点は正、細部を捏造することがある）
- 宣言文への engage がやや平板（acknowledge 止まり）
- 既知の瑕疵 #web1: 直前 Web 結果のメモが「関連だが別の質問」を横取りし得る
  （例: Canberra をlookup後の「人口は?」が再検索せず捏造）。要調整。

---

## 6. 既知の限界・注意
- 1.5B distill のため多段算術・細部事実は限界あり（より大きいモデル/Web優先で緩和可）。
- Web は DuckDuckGo→Wikipedia スクレイピング（キー不要）。レート制限・HTML変更で不安定な場合あり。
- Web ターンは取得込みで 60–100s。
- 長会話では SP 圧縮で具体値が劣化し得る（明示保存値は recall で復元可）。

---

## 7. 再現とトレース
- パッチ単体: `cotscan_live/faithful_patches/app_session_torch.patch`
- バッテリ: `conv_battery.py`(雑談/既知/算術/想起/創作) / `conv_web.py`(複合 recall×web) / `conv_math.py`(多段算術診断)
- 経緯: `STATUS_2026-06-16_faithful_iteration.md`
