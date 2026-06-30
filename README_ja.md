# ComfyUI-LoRA-Merge-EXPERIMENTAL

[English README](README.md)

[NP-LoRA](https://arxiv.org/html/2511.11051v3) と [SSR-Merge](https://arxiv.org/abs/2606.10617) に対応した、ComfyUI向け LoRA マージ用カスタムノード集です。

現在は次のノードを含みます。

- **NP-LoRA Loader (Subject + Style)**
  - 被写体/キャラクター LoRA とスタイル LoRA を、NP-LoRA の非対称 null-space projection で合成して適用します。
- **SSR-Merge Calibration**
  - SSR-Merge 用に1回だけキャリブレーション推論を行い、内部特徴の統計から通常LoRA形式のマージ結果を作ります。
- **SSR-Merge Loader**
  - `SSR-Merge Calibration` の結果を `MODEL` に適用します。

内部実装は、現在の ComfyUI カスタムノードAPIに合わせています。

## インストール

このディレクトリを ComfyUI の `custom_nodes` に配置します。現在の配置からは次で導入できます。

```powershell
Copy-Item -Recurse D:\tools\dev\ComfyUI-LoRA-Merge-EXPERIMENTAL D:\tools\ComfyUI\ComfyUI\custom_nodes\
```

ComfyUI を再起動後、`loaders/LoRA` から各ノードを追加してください。

## NP-LoRA Loader

NP-LoRA 論文の Eq. 12 に従い、スタイル LoRA の右特異ベクトル空間から content LoRA の重なる成分を連続的に減衰させます。

`D_merged = D_style + D_content (I - mu/(1+mu) V V^T)`

基本的な使い方:

1. `content_lora` に被写体・人物・キャラクターなどの LoRA を指定します。
2. `style_lora` に画風 LoRA を指定します。
3. 通常は `mu = 0.5` から始めます。
4. スタイルが弱い場合は `mu` を上げ、被写体の細部が失われる場合は下げます。`mu = 0` は通常の加算マージです。

## SSR-Merge

SSR-Merge は、マージ前に短いキャリブレーション推論を行い、その内部特徴からルータを解析的に求めます。ルータは up-projection 側に吸収されるため、最終的には通常の LoRA として適用できます。

ComfyUI では次の2ノード構成にしています。

1. **SSR-Merge Calibration**
   - `model` / `clip`
   - `lora_1` / `lora_2`
   - `prompt_1` / `prompt_2`
   - `negative_prompt`
   - seed、解像度、`lambda_reg`
   - sampler / scheduler

2. **SSR-Merge Loader**
   - `SSR-Merge Calibration` の `ssr merge` 出力を受け取り、`MODEL` に適用します。

### キャリブレーション動作

`SSR-Merge Calibration` は、ノード実行時に新しくキャリブレーションを行います。

キャリブレーション推論の steps と cfg は軽量化のため固定です。

- steps: `1`
- cfg: `1.0`

sampler と scheduler はノードUIから指定できます。

SSR-Merge は正常処理時も ComfyUI コンソールへ `INFO` ログを出します。主に次を確認できます。

- LoRAごとの抽出レイヤー数
- SSR対象レイヤー数と passthrough レイヤー数
- 各LoRAのキャリブレーション開始・終了
- 登録hook数
- 統計取得済みレイヤー数
- router solve の完了状況
- Loaderの処理時間

### 現在の対応範囲

- U-Net/DiT 側の通常 LoRA
- 線形層と 1x1 畳み込み LoRA
- 2つの LoRA の SSR マージ

CLIP側LoRA、LoCon の空間カーネル、DoRA、LoHa、LoKr などは安全のため SSR 対象外です。

## 参照

- [NP-LoRA: Null Space Projection for Subject-Style LoRA Fusion](https://arxiv.org/html/2511.11051v3)
- [SSR-Merge: Subspace Signal Routing for Training-Free LoRA Merging in Diffusion Models](https://arxiv.org/abs/2606.10617)
- [SSR-Merge official implementation](https://github.com/nagara214/SSR-Merge)
