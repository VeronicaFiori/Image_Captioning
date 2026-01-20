# mini_lavis_flickr8k (BLIP2-style, scritto da zero)

Repo pronto per image captioning su Flickr8k con architettura:

OpenCLIP ViT (frozen) -> Q-Former (trainable) -> Flan-T5 (unico LLM)

Include anche valutazione fattuale opzionale con Qwen2-VL-7B-Instruct.

## 1) Dataset: struttura cartelle
Metti Flickr8k cosi (nomi file esatti):

```
mini_lavis_flickr8k/
  data/
    flickr8k/
      Images/
        1000268201_693b08cb0e.jpg
        ...
      Flickr8k.token.txt
      Flickr_8k.trainImages.txt
      Flickr_8k.devImages.txt
      Flickr_8k.testImages.txt
```

## 2) Install
Da dentro la cartella del progetto:

```
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

## 3) Training (Stage A: veloce)
Esempio (usa config di default per Flickr8k):

```
python -m src.train.trainer \
  --config configs/flickr8k_caption.yaml \
  --data_root data/flickr8k \
  --out_dir outputs/flickr8k_stageA
```

Il checkpoint migliore (val loss) viene salvato in:
`outputs/flickr8k_stageA/checkpoints/best.pt`

## 4) Inference su singola immagine

```
python -m src.infer.generate \
  --ckpt outputs/flickr8k_stageA/checkpoints/best.pt \
  --image data/flickr8k/Images/<img>.jpg \
  --prompt "Describe the image in one sentence." \
  --num_beams 5
```

## 5) Demo Gradio

```
python demo/app_gradio.py \
  --ckpt outputs/flickr8k_stageA/checkpoints/best.pt
```

## 6) Valutazione
### 6.1 Metriche classiche (BLEU con sacrebleu)

```
python -m src.eval.caption_metrics \
  --config configs/flickr8k_caption.yaml \
  --data_root data/flickr8k \
  --ckpt outputs/flickr8k_stageA/checkpoints/best.pt \
  --split test
```

### 6.2 Qwen2-VL judge (opzionale, pesante)
Serve molta VRAM oppure quantizzazione 4-bit.

```
python -m src.eval.qwen_vl_judge \
  --data_root data/flickr8k \
  --preds outputs/flickr8k_stageA/preds_test.json \
  --out outputs/flickr8k_stageA/qwen_judge.json \
  --load_in_4bit
```

Suggerimento: fai girare Qwen judge su una GPU >= 24GB se puoi, oppure 16GB con 4-bit.

## 7) Script rapidi Atlernativi
- `scripts/train_caption.sh`
- `scripts/infer_caption.sh`
- `scripts/eval_all.sh`

# Image_Captioning
