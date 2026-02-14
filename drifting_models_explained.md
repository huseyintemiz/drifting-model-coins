# Drifting Models: Teknik Dokumantasyon

Bu dokuman, "Generative Modeling via Drifting" (Deng et al., 2026) paper'inin bu repodaki implementasyonunu Turkce-Ingilizce karisik teknik bir dille aciklamaktadir.

Paper: https://arxiv.org/abs/2602.04770

---

## 1. Core Idea: Neden Drifting?

Klasik generative modeller (Diffusion, Flow Matching) **inference time**'da iterasyon yapar — sample uretirken bircok step gerekir. Drifting Models tam tersini yapiyor: **iterasyonu training time'a** tasiyor. Sonuc olarak inference'da tek bir forward pass yeterli oluyor (1-NFE, one network function evaluation).

## 2. Training Objective

Generator `f_theta`, noise `epsilon`'dan sample uretir. Training loss'u:

```
L = E[ || f_theta(eps) - stopgrad(f_theta(eps) + V) ||^2 ]
```

Adimlar:
1. Generator noise'dan bir sample uretiyor: `gen = f_theta(eps)`
2. **Drift field V** hesaplaniyor — generated sample'in nereye "kaymasi" gerektigini soyluyor
3. Target = `gen + V` (yani "biraz kaydirilmis hali")
4. Loss = generated ile target arasindaki MSE

`stopgrad` kritik — gradient sadece generator'a gidiyor, V'ye degil. Yani generator her step'te "biraz daha dogru yere uretmeyi" ogreniyor.

Implementasyon (`drifting_model_demo.ipynb`):

```python
def drifting_loss(gen, pos, compute_drift):
    with torch.no_grad():
        V = compute_drift(gen, pos)
        target = (gen + V).detach()
    return F.mse_loss(gen, target)
```

## 3. Drift Field V: Mean-Shift Tabanli Hesaplama

Her generated point icin, "real data'ya dogru cekilis" ile "diger generated point'lerden itilis" arasindaki fark hesaplaniyor. Mean-shift'ten esinlenmis:

```
V(x) = V+(x) - V-(x)
```

- **V+ (positive/attractive):** Real data sample'larina (pos) dogru cekiyor — "gercek data burada, oraya git"
- **V- (negative/repulsive):** Diger generated sample'lardan uzaklastiriyor — "orada zaten baska bir generated var, mode collapse yapma"

### Implementasyon Adimlari (Algorithm 2)

Dosya: `drifting.py:11-73`, `compute_V()` fonksiyonu.

1. `torch.cdist` ile generated ve tum sample'lar arasi **L2 distance** hesaplaniyor
2. **Self-distance**'lar maskeleniyor (kendine olan mesafe = 1e6, cunku y_neg = x)
3. Logit'ler: `logit = -dist / temperature`
4. **Cift yonlu normalizasyon** (paper'in key insight'i): hem row hem column uzerinden softmax alinip geometric mean hesaplaniyor: `A = sqrt(softmax_row * softmax_col)`. Bu batch size'a bagimliliga azaltiyor
5. **Cross-weighting:** `W_pos = A_pos * sum(A_neg)`, `W_neg = A_neg * sum(A_pos)`
6. Final drift: `V = W_pos @ y_pos - W_neg @ y_neg`

### Neden V=0 Olunca Training Bitiyor?

Cok elegant bir property: Eger generated distribution `q` ile real distribution `p` ayniysa, formuludeki `(y+ - y-)` terimi **anti-symmetric** oluyor ve expectation'da cancel oluyor. Yani `V = 0` → loss = 0 → convergence.

### Temperature (tau) Ne Ise Yariyor?

`temp` parametresi kernel'in "ne kadar local" oldugunu kontrol ediyor:

| Temperature | Etki | Yakaladigi Yapi |
|-------------|------|-----------------|
| 0.02 (kucuk) | Sadece cok yakin noktalari etkiler | Fine-grained detail |
| 0.05 (orta) | Orta mesafe neighborhood | Mid-level structure |
| 0.20 (buyuk) | Genis bir neighborhood | Global structure |

Bu yuzden actual training'de (`drifting.py:76-113`) **multi-temperature** kullaniliyor. Her temperature'dan gelen V ayri ayri normalize edilip toplaniyor. Boylece hem local hem global structure ayni anda ogreniyor.

```python
# drifting.py — compute_V_multi_temperature()
for tau in [0.02, 0.05, 0.2]:
    V_tau = compute_V(x, y_pos, y_neg, tau)
    V_tau = V_tau / sqrt(mean(V_tau^2))   # normalize
    V_total += V_tau
```

## 4. Toy 2D'den Image Generation'a Gecis

Toy 2D'de basit bir MLP vardi: `noise (32-D) → MLP → 2D point`. Actual image generation'a geciste uc major fark var.

### 4a. Generator: DriftDiT (Diffusion Transformer)

Dosya: `model.py`, `DriftDiT` class'i.

Standard DiT'ten farklari:
- **Timestep yok** — one-step generator, diffusion degil
- **Conditioning** = label + alpha + style (3 embedding toplami)
- Register tokens, RoPE, SwiGLU, RMSNorm, QK-Norm kullanir

**Forward pass** (`model.py:458-508`):

```
Noise eps (B, C, 32, 32)
    |
    v
PatchEmbed (Conv2d, stride=4)      # 32x32 image → 8x8 = 64 patch, her biri hidden_dim-D
    |
    v
[8 register token] ++ [64 patch token] = 72 token sequence
    |
    v
Conditioning: c = label_embed + alpha_embed + style_embed
    |
    v
N x DiTBlock                        # her block: adaLN-modulated attention + SwiGLU MLP
    |
    v
Register tokens cikarilir → 64 patch token kalir
    |
    v
FinalLayer → Unpatchify → Image (B, C, 32, 32)
```

Iki model varianti:

| | DriftDiT-Tiny | DriftDiT-Small |
|---|---|---|
| Depth | 6 block | 8 block |
| Hidden dim | 256 | 384 |
| Heads | 4 | 6 |
| Kullanim | MNIST | CIFAR-10, TurCoins |

### 4b. Conditioning Mekanizmasi

Uc embedding toplanip `c` vektoru olusturuyor. Her DiTBlock bu `c`'den 6 modulation parametresi uretiyor.

**LabelEmbedder** (`model.py:257-284`): Class label'i embed ediyor. Training'de %10 ihtimalle label'i "null class"a drop ediyor — bu CFG icin gerekli. `num_classes + 1` embedding var, son index = unconditional.

**AlphaEmbedder** (`model.py:287-312`): CFG scale alpha'yi Fourier features ile encode ediyor. Neden? Cunku inference'da farkli alpha degerleri kullanabilmek istiyoruz ve model'in alpha'ya gore davranisini ogrenmesi lazim.

**StyleEmbedder** (`model.py:315-336`): 64 token'lik bir codebook'tan random 32 token sec, embed et, topla. Bu diversity sagliyor — ayni class ve alpha ile farkli style'lar uretebiliyorsun. Her forward pass'ta farkli random token secildigi icin her sample farkli oluyor.

**adaLN-Zero** (`model.py:151-207`): Her DiTBlock, conditioning `c`'den 6 parametre uretiyor:

```
c → SiLU → Linear → (shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp)
```

- `shift/scale`: RMSNorm'dan sonra feature'lari modulute ediyor
- `gate`: Residual connection'i kontrol ediyor (0 = skip, 1 = full contribution)

**Zero initialization trick** (`model.py:429-432`): Modulation layer'larin weight'leri sifirla initialize ediliyor. Training basinda gate = 0, block'lar identity gibi davraniyor. Bu training stability icin kritik.

### 4c. Feature Space: Pixel vs ResNet Features

Drift field V dogrudan pixel space'te veya learned feature space'te hesaplanabiliyor. Secim dataset'e bagli:

**MNIST — Pixel space** (`train.py:224-227`):
```python
feat = x.flatten(start_dim=1)  # (B, 1*32*32) = (B, 1024)
```
Image'i duzlestir, 1024-D vector olarak kullan. MNIST yeterince basit oldugu icin bu calisiyor.

**CIFAR-10 / TurCoins — Pretrained ResNet18** (`feature_encoder.py:16-64`):
```
x → (resize to 64x64) → conv1 → bn → relu → maxpool
  → layer1 → f1 (64-ch)
  → layer2 → f2 (128-ch)
  → layer3 → f3 (256-ch)
  → layer4 → f4 (512-ch)
return [f1, f2, f3, f4]   # 4 farkli scale'de feature map
```

ImageNet-pretrained ResNet18 **frozen** olarak kullaniliyor (gradient almiyor, sadece feature extractor). Neden pretrained? Pixel space'te drifting CIFAR gibi complex image'larda calismiyor — semantik anlamli (meaningful) bir feature space lazim.

**Multi-scale loss** (`train.py:234-284`, Paper Section A.5):

```python
# Her scale'deki feature map → global average pool → vector
feat_gen_list = [F.adaptive_avg_pool2d(f, 1).flatten(1) for f in feat_gen_maps]
# feat_gen_list = [64-dim, 128-dim, 256-dim, 512-dim]
```

Her scale icin ayri ayri V hesaplanip loss toplaniyor. Low-level features (f1, 64-ch) texture/edge gibi seyleri yakalarken, high-level features (f4, 512-ch) semantic content'i yakaliyor.

## 5. Training Pipeline

Dosya: `train.py`, `train_step()` fonksiyonu (satir 298-370).

Her training step'te:

```
1. 10 random class sec                           (batch_nc = 10)
2. Her class icin 32 noise sample uret            (n_neg = 32, toplam 320)
3. alpha ~ Uniform(1.0, 3.0) sample et
4. Generator: G(noise, labels, alpha) → 320 generated image
5. SampleQueue'dan her class icin 32 real sample cek  (n_pos = 32)
6. Her class AYRI AYRI:
   a. Feature'lari hesapla (pixel veya ResNet multi-scale)
   b. L2 normalize (unit sphere'e project et)
   c. Her temperature icin V hesapla, normalize et, topla
   d. MSE loss hesapla
7. Tum class loss'larini ortala
8. Backward + gradient clip (max norm 2.0) + optimizer step
9. EMA update (decay 0.999)
10. Learning rate warmup schedule
```

**SampleQueue** (`utils.py`) neden var? Standard DataLoader'dan gelen batch'lerde her class'tan yeterli sample olmayabilir. Queue her class icin 128 sample cache'liyor, her step'te oradan random 32 cekiyor.

**Neden per-class V?** Drift field'in mantikli olmasi icin pos sample'larin ayni class'tan olmasi lazim. "3" uretmeye calisan bir generated sample'i "7"ye dogru drift ettirmek istemezsin.

## 6. Inference: Classifier-Free Guidance (CFG)

Dosya: `model.py:510-550`, `forward_with_cfg()`.

Inference'da tek bir forward pass yeterli (1-NFE), ama CFG icin iki forward pass yapilir:

```python
# Ayni noise'u iki kere forward pass yap:
# 1) Conditional:   label = gercek class
# 2) Unconditional: label = null class (force_drop = True)
output = uncond + alpha * (cond - uncond)
```

Alpha buyudukce class conditioning gucleniyor:
- `alpha = 0`: Tamamen unconditional
- `alpha = 1`: Balanced
- `alpha = 1.5`: Genelde sweet spot
- `alpha = 3+`: Over-saturation riski

## 7. Dataset'lere Ozel Detaylar

### MNIST
- Resolution: 28×28 → resize to 32×32
- 1 channel (grayscale), 10 class
- Feature space: Pixel (1024-D flat vector)
- Model: DriftDiT-Tiny, ~100 epoch, ~20 dk GPU'da

### CIFAR-10
- Resolution: 32×32 native
- 3 channel (RGB), 10 class
- Feature space: Pretrained ResNet18, 4-scale
- Model: DriftDiT-Small, 200 epoch
- Augmentation: Random horizontal flip

### TurCoins
- Source: HuggingFace `hsyntemiz/turcoins`
- Resolution: Variable → resize + crop to 32×32
- 3 channel (RGB), 138 class
- Feature space: Pretrained ResNet18, 4-scale
- Model: DriftDiT-Small, 200 epoch
- Evaluation: Ayri bir ViT-B/16 classifier (`train_classifier.py`)

## 8. Toy 2D vs Full Pipeline Karsilastirmasi

| | Toy 2D | Image Generation |
|---|---|---|
| Generator | MLP (32-D → 2-D) | DriftDiT (noise → 32×32 image) |
| Feature space | Dogrudan output (2D) | Pixel (MNIST) veya ResNet multi-scale (CIFAR/TurCoins) |
| Conditioning | Yok | Label + Alpha + Style (adaLN-Zero) |
| V computation | Tek temperature, tum data birlikte | Multi-temp, per-class, per-scale |
| Inference | Tek forward pass | CFG: 2 forward pass (conditional + unconditional) |
| Normalization | Yok | L2 norm + V norm + feature standardization |

## 9. Dosya Yapisi ve Sorumluluklar

```
model.py            — DriftDiT generator (PatchEmbed, DiTBlock, adaLN-Zero, RoPE, CFG)
drifting.py         — compute_V(), multi-temperature, normalization, loss class'lari
feature_encoder.py  — PretrainedResNetEncoder (CIFAR/TurCoins), MultiScaleFeatureEncoder (MNIST)
train.py            — Training loop, config dict'ler, SampleQueue management, per-class batching
sample.py           — One-step inference, grid visualization, alpha sweep, FID computation
utils.py            — EMA, WarmupLRScheduler, SampleQueue, checkpoint save/load
train_classifier.py — TurCoins evaluation icin ViT-B/16 classifier
```
