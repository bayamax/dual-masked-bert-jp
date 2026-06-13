# TRIGGER_V1 結果 (2026-06-13)

リコールトリガー実験第一段: 「フルKV教師のアテンションが露出窓(512tok)より前の
evicted領域に飛ぶか」を質問種別に計測し、生徒特徴(質問位置pre-RoPEクエリ、
Phase Bと同一採取)からそれを予測するヘッドを訓練した。

実行: RunPod A6000、TRIGGER_POD_V1、00:21–00:47 UTC(26分)、実費 $0.156。
数値は回収した npz からローカル再計算で検証済み(ログ転記ではない)。

## 前提検証(教師アテンションの行き先、BOSシンク除外、n=1200 train / 360 heldout)

evicted質量比の中央値(train、層 8/14/20 = L0/L1/L2):

| 質問種別 | L0 | L1 | L2 |
|---|---|---|---|
| EVICT(窓外needle想起) | .371 | **.490** | .413 |
| WINDOW(窓内needle想起・ハード陰性) | .181 | .130 | .118 |
| GENERIC(過去参照なし) | .266 | .236 | .127 |

- EVICT p10 (L1=.439) > WINDOW p90 (L1=.151): **分布が完全分離**。
  測定値そのものの ORACLE AUC = 1.0000(evict-vs-rest / evict-vs-window とも)。
- 「必要なときだけ窓外にアテンションが飛ぶ」という前提は合成バッテリー上で成立。
- GENERIC でも .13–.27 の拡散質量が窓外に飛ぶ(長文履歴への背景アテンション)
  ため、生の質量に固定閾値を切るなら層と閾値の選択が要る(L1 が最も分離)。

## トリガーヘッド(ロジスティック回帰 4608dim、C=0.003、GroupKFold by seed)

評価は未知シード+未知テンプレ(heldout 5種、n=360):

- EVAL AUC evict-vs-all: **1.0000**
- EVAL AUC evict-vs-window(本丸): **1.0000**
- EVAL AUC evict-vs-generic: **1.0000**

既存 attn_trigger4(AUC 0.830、v1 で OFF)に対し、ラベルをフルKV実アテンションに
変えただけで合成バッテリー上は天井。重み: results/trigger_head.npz。

## 注意(まだ言えないこと)

1. **合成テンプレ域での天井**であり、実会話・実 web 混在ターンへの汎化は未測。
   composite 系バッテリーでの実戦評価が次。
2. 生徒特徴はフルKV履歴キャッシュ下で採取(Phase B と同形 = 上限実証の扱い)。
   本番は SP+露出窓文脈なので、V3 と同じ形で SP 文脈下の特徴で再採取→再学習が必要。
3. WINDOW 陰性は「直近に植えた needle」なので、recency が手掛かりになっている
   可能性がある。ただし本番でも「窓内=直近」なので致命的ではない。

## ファイル

- trigger_labels.py / trigger_train.py / run_trigger.sh — 生成・訓練・ジョブ
- results/trigger_labels_{0,1,999}.npz — ラベル(train/heldout/smoke)
- results/trigger_head.npz — ヘッド重み(coef/intercept/scaler)
- results/*.log, results/STATUS.txt — Pod 実行ログ一式

# TRIGGER_V2 / V3 結果 (2026-06-13 追記)

## V2: 本番文脈(SP+露出窓)での再採取 — results_sp/
[BOS][SP(pooler圧縮)][露出窓512][質問] の本番レイアウトで特徴を採り直して再学習。
A4000、9分、$0.026。npzからローカル再計算で検証済み。
- EVAL AUC: evict-vs-all / -window / -generic 全て **1.0000**
- **V1ヘッド(フルKV文脈学習)の転移も1.0000** — 文脈ズレは検索キーには致命的
  (Phase A実測)だがトリガー判定には効かない。
- 本体SFT不要・本番投入可能な形の判定器が確定。

## V3: テンプレ拡張+LOOKUP陰性クラス — results_v3/
gpt-4o-mini生成+機械検証のテンプレ拡張(needle 15+52 / generic 8+25 /
filler 16+25、評価シャードは全カテゴリ未知テンプレ)+web行き質問39種を
4クラス目の陰性として追加。A5000、8分、$0.022。
- V3ヘッド EVAL AUC: evict-vs-all .9998 / **-window .9994** / -generic 1.0000 /
  **-lookup .9997** — テンプレ多様性を大きく広げてもセッション内トリガーは天井近くを維持。
- V2ヘッド(lookup陰性なしで学習)の転移: -window .9984 / -generic .9965 は保つが
  **-lookup .8063 に崩落** — 「過去参照とweb参照の混同」というアプリの実症状を
  バッテリー上で定量再現。lookup陰性を訓練に入れるだけで .9997 に回復。
- 採用ヘッド: results_v3/trigger_head_v3.npz(4608dim ロジスティック回帰)。

## 次: アプリ配線(学習なし)
毎ターン無条件 retrieve → 「ヘッド発火ターンのみ retrieve」(SPCHAT_TRIGGER=head)。
評価: needle想起維持 + v1/v2回帰 + 引き戻しスキップ率。
