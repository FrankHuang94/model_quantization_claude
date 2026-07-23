# 04. Precision Formats and Numerics

> **Section scope.** The numeric *representations* that quantization targets: the
> integer formats (INT16/8/4/2, binary, ternary), the reduced-precision
> floating-point formats (FP16, BF16, FP8, FP6, FP4), block floating point and the
> microscaling (MX) family, and the non-uniform formats (NF4, logarithmic,
> codebook). Section 03 covered the *techniques* that map high-precision tensors
> into these formats; this section characterizes the formats themselves — their
> dynamic range, precision, hardware support, and best-fit use cases — because the
> choice of format is increasingly a hardware decision that constrains everything
> upstream.

## Two families: integer and floating point

Every numeric format used in quantization belongs to one of two families, and the
distinction between them is the organizing idea of this section.

**Integer (fixed-point) formats** represent values as evenly-spaced points. An
`n`-bit integer format has `2ⁿ` levels spread *uniformly* across a range set by an
external scale factor (Section 03's affine mapping). The defining property is
**uniform resolution**: the gap between adjacent representable values is constant
across the whole range. This is efficient when the data occupies the range fairly
evenly, and it maps to the cheapest possible hardware — integer multiply-accumulate
units are small and energy-efficient. The weakness is dynamic range: to represent
both very large and very small magnitudes, an integer format must either use a coarse
step (losing small values) or a narrow range (clipping large ones).

**Floating-point formats** represent values as `sign × mantissa × 2^exponent`,
splitting the bits between a mantissa (precision) and an exponent (range). The
defining property is **relative resolution**: the gap between adjacent values scales
with the magnitude, so small numbers get fine steps and large numbers get coarse
steps — a logarithmic-ish spacing that matches how neural-network values are actually
distributed (many small, few large). Floating point handles outliers gracefully (a
large value just uses a large exponent) at the cost of more complex hardware and,
for a fixed bit budget, fewer bits available for the mantissa.

The entire "INT vs FP" debate at low bit-widths — INT4 vs FP4, INT8 vs FP8 — is a
debate about whether uniform or relative resolution better fits a given tensor's
distribution, traded against hardware cost. There is no universal winner, which is
why modern silicon increasingly supports *both* and lets the compiler choose.

## The range-versus-precision tradeoff, visualized

![Numeric format map: dynamic range vs. precision](../assets/charts/04_format_range_precision.png)

The format map plots each format's dynamic range (how many orders of magnitude of
value it can represent, horizontal) against its precision (effective mantissa bits,
vertical), with bubble size proportional to total bit width. The structure is
revealing. The **integer formats** (blue) sit at low dynamic range but high
uniform-precision-for-their-bits — INT8 gives a full 8 bits of uniform resolution
but only ~2.4 orders of magnitude of range (set by its scale). The **floating-point
formats** (red) trade precision for range: FP8-E5M2 reaches ~9.5 orders of magnitude
but with only 2 mantissa bits, while FP8-E4M3 keeps 3 mantissa bits at ~5.7 orders.
**BF16** is the extreme range choice (8 exponent bits give it the same ~38-order
range as FP32) at the cost of only 7 mantissa bits — which is exactly why it won for
training, where gradient dynamic range matters more than mantissa precision. **NF4**
(green) sits apart: a non-uniform 4-bit format that gets better *effective* precision
than uniform INT4 by placing its levels where the data is dense.

The practical reading: **choose integer when the data is range-limited and you want
maximum uniform precision per bit (weights, post-normalization activations); choose
floating point when the data has outliers or wide dynamic range (raw activations,
gradients); and choose the exponent/mantissa split to match whether range or
precision is your binding constraint.**

## Integer formats in detail

### INT8 — the workhorse

INT8 (8-bit signed or unsigned integer) is the most important quantization format in
history and remains the production baseline. With 256 levels and, in practice,
per-channel or per-group scaling, INT8 is effectively lossless for the vast majority
of models: vision CNNs lose well under 1% top-1, and well-calibrated LLMs lose 1–2
perplexity-equivalent points at most. Every mobile NPU, every inference engine
(TFLite, TensorRT, Core ML, OpenVINO, QNN), and every training framework supports it.
Its ubiquity means INT8 hardware is mature, fast, and cheap, and its accuracy is
well-understood. INT8 is the format you use unless you have a specific reason not to.

### INT16 — the safety margin

INT16 is used where INT8 loses too much accuracy but a full float is unnecessary —
some audio DSP paths, certain sensitive layers in a mixed-precision model, and
accumulation. It is rarely the *primary* deployment format (if you can afford 16 bits
you often just use FP16), but it exists as an intermediate and as an accumulator
width. Many NPUs support INT16 activations paired with INT8 weights (W8A16) as a
higher-accuracy option.

### INT4 — the LLM weight format

INT4 (16 levels) became the defining format of the on-device-LLM era. Naive
per-tensor INT4 is far too coarse for general use, but **per-group INT4** (group size
128 or 64, with an FP16 or INT8 scale per group) is the workhorse of weight-only LLM
quantization — the target of GPTQ, AWQ, and the GGUF k-quants. At 4 bits weight-only,
LLMs retain most of their capability while cutting weight memory ~4×, which is the
single most important compression result in edge AI. INT4 is now natively supported
in most flagship mobile NPUs (Qualcomm since Snapdragon 8 Gen 2, and others),
NVIDIA tensor cores, and Intel/AMD accelerators. For *activation* quantization, INT4
(W4A4) is much harder and remains research-stage (Section 05).

### INT2 and ternary — the frontier

INT2 (4 levels) and ternary ({−1, 0, +1}, ≈1.58 bits) are the aggressive frontier.
Post-training INT2 of a normally-trained model generally collapses accuracy unless
paired with sophisticated methods (QuIP#, AQLM codebooks) that approach the
information-theoretic limit — and even then the decoding kernels are complex.
Ternary's success story is different: **quantization-native training** (BitNet
b1.58) trains the model with ternary weights from scratch, and such models match
full-precision quality because the weights are *learned* to be ternary rather than
forced. INT2 arrived in shipping silicon with the Snapdragon 8 Elite Gen 5 (2025)
⚠️ — hardware provisioning ahead of broad software use.

### Binary — the extreme

1-bit binary weights (and, in BNN/XNOR-Net, binary activations) are the theoretical
floor. They enable replacing multiplies with XNOR+popcount for enormous energy
savings, but the accuracy gap on non-trivial tasks has kept them research-only for a
decade. Binary remains interesting for ultra-low-power always-on tinyML sensing
(keyword spotting, wake-word, simple vision) where the task is easy enough to
tolerate the accuracy loss and the energy budget is measured in microjoules.

| Integer format | Levels | Typical granularity | Maturity | Primary use |
|---|---|---|---|---|
| INT16 | 65,536 | per-tensor/channel | 🟢 | Sensitive layers, accumulation, audio |
| INT8 | 256 | per-channel | 🟢 | Universal baseline (vision, W8A8) |
| INT4 | 16 | per-group (g=128/64) | 🟢 (weight-only) | On-device LLM weights |
| INT2 | 4 | per-group + codebook | 🟡 silicon / 🔴 general SW | Frontier; needs advanced methods |
| Ternary (1.58b) | 3 | per-tensor (native-trained) | 🔴 research | Quantization-native (BitNet) |
| Binary (1b) | 2 | per-filter scale | 🔴 research | tinyML always-on sensing |

## Floating-point formats in detail

### FP32, FP16, BF16 — the high-precision baselines

FP32 (1 sign / 8 exponent / 23 mantissa) is the training default of the past and the
reference against which quantization loss is measured. **FP16** (1/5/10) halves the
storage with 10 mantissa bits and a modest exponent range, and is the standard
*activation* precision in weight-only-quantized LLMs (the "A16" in W4A16). **BF16**
(1/8/7) keeps FP32's full exponent range but only 7 mantissa bits; the wide range
makes it far more robust for training (gradients span many orders of magnitude), and
BF16 is now the dominant training format. The FP16-vs-BF16 choice is the cleanest
illustration of the range/precision tradeoff: same 16 bits, different split, chosen
by whether you need mantissa precision (FP16, inference activations) or exponent range
(BF16, training).

### FP8 — the data-center-to-edge format

FP8 comes in two OCP-standardized variants: **E4M3** (1/4/3, range to ±448, better
precision) and **E5M2** (1/5/2, wider range ~±57,344, less precision). The convention
is E4M3 for forward-pass activations and weights (precision matters) and E5M2 for
gradients (range matters). FP8 is production in the data center (NVIDIA Hopper/
Blackwell, and others) and is arriving on flagship mobile NPUs (Snapdragon 8 Elite
Gen 5 lists FP8). FP8's advantage over INT8 is graceful outlier handling — the
exponent absorbs large values that would force an INT8 scale to coarsen — which makes
FP8 attractive for *activation* quantization of transformers where INT8 struggles.
Its disadvantage is more complex hardware and, at 3 mantissa bits (E4M3), less
precision than INT8's uniform 8 for range-limited data. FP8 is the format most likely
to become a second edge baseline alongside INT8.

### FP6, FP4, and the sub-8-bit floats

Below 8 bits, floating point continues: **FP6** (e.g. E3M2) and **FP4** (E2M1, 1 sign
/ 2 exponent / 1 mantissa) push the range/precision tradeoff to its limit. FP4 has
only *sixteen* representable values — the same count as INT4 — but distributed with
floating-point (roughly logarithmic) spacing rather than uniformly, which better
matches weight distributions and handles outliers, at the cost of only 1 mantissa
bit. FP4 is the frontier format: NVIDIA Blackwell (2024–2025) brought native FP4
tensor cores and demonstrated FP4 *training*, and the newest mobile NPUs are
beginning to list FP4 support ⚠️. The open question, as with INT2, is software
maturity — the hardware exists ahead of accuracy-preserving FP4 quantization
toolchains.

| Float format | Bits (S/E/M) | Approx range | Maturity | Primary use |
|---|---|---|---|---|
| FP32 | 1/8/23 | ±3.4e38 | 🟢 | Reference / legacy training |
| BF16 | 1/8/7 | ±3.4e38 | 🟢 | Training default; wide-range |
| FP16 | 1/5/10 | ±65,504 | 🟢 | Inference activations (A16) |
| FP8 E4M3 | 1/4/3 | ±448 | 🟢 DC / 🟡 edge | Fwd activations/weights |
| FP8 E5M2 | 1/5/2 | ±57,344 | 🟢 DC / 🟡 edge | Gradients; wide-range activations |
| FP6 E3M2 | 1/3/2 | ±28 | 🟡 | Emerging middle ground |
| FP4 E2M1 | 1/2/1 | ±6 | 🟡 DC / 🔴 edge SW | Frontier; Blackwell + new NPUs |

## Block floating point and the microscaling (MX) formats

The most important recent development in numeric formats is **block floating point
(BFP)**, standardized by the Open Compute Project as the **Microscaling (MX)**
family. The idea unifies the integer and floating-point families and bakes Section
03's per-group quantization directly into the numeric type.

In a microscaling format, a **block** of `k` elements (the OCP MX standard fixes
`k = 32`) shares a single **scale** — specifically a shared 8-bit exponent (E8M0) —
while each element carries a small low-precision value (FP or INT). The named types
are **MXFP8, MXFP6, MXFP4** (block of 32 FP8/6/4 values with a shared scale) and
**MXINT8** (block of 32 INT8 values with a shared scale). This is exactly per-group
quantization with group size 32, but promoted from a software convention to a
hardware-native numeric format with a standardized layout, so the hardware can
consume it directly without the compiler having to manage scales separately.

Why this matters:

- **It closes the algorithm–hardware gap.** Per-group quantization was the software
  technique that made 4-bit work (Section 03); MX makes the *hardware* natively
  understand per-group scales, so there is no mismatch between what the algorithm
  wants and what the silicon executes. The block size 32 is chosen to match a
  hardware-natural tile.
- **It gets the best of both families.** Within a block, the shared exponent gives
  floating-point-like dynamic range (outliers in the block are absorbed by the shared
  scale), while the elements can be cheap low-precision values. A block of MXFP4 has
  effectively per-32 dynamic-range adaptation, which is far more robust than plain
  FP4.
- **It has broad industry backing.** The MX formats were standardized (2023–2024) by
  a consortium including AMD, Arm, Intel, Meta, Microsoft, NVIDIA, and Qualcomm —
  unusual cross-vendor agreement that signals MX is the likely convergence point for
  sub-8-bit hardware numerics. NVIDIA Blackwell implements MX (and a variant,
  NVFP4); mobile vendors are expected to follow.

Block floating point is not new — DSPs have used it for decades, and it appeared in
various AI accelerators — but the OCP standardization and its adoption by the major
silicon vendors make MX the format story to watch. In effect, the field's per-group
quantization technique and its numeric-format evolution have merged.

| MX format | Element type | Block | Shared scale | Effective bits/elem | Status |
|---|---|---|---|---|---|
| MXINT8 | INT8 | 32 | E8M0 | ~8.25 | 🟡 emerging |
| MXFP8 | FP8 (E4M3/E5M2) | 32 | E8M0 | ~8.25 | 🟡→🟢 (Blackwell) |
| MXFP6 | FP6 | 32 | E8M0 | ~6.25 | 🟡 |
| MXFP4 | FP4 (E2M1) | 32 | E8M0 | ~4.25 | 🟡 (Blackwell / NVFP4) |

## Non-uniform and companded formats

As Section 03 discussed, neural weights are bell-shaped, and *non-uniform* formats
that place levels to match this distribution can beat uniform formats at equal bits.

- **NF4 (NormalFloat4)** places its 16 levels at the quantiles of a standard normal
  distribution, so — for the roughly-Gaussian weights of a neural network — each
  level is used about equally often, which is information-theoretically efficient.
  NF4 is the most successful non-uniform format in production (via QLoRA and the
  bitsandbytes library) and often beats plain INT4 at the same 4 bits. A variant,
  **AF4 (AbnormalFloat)**, tunes the levels to the empirical (not assumed-normal)
  weight distribution.
- **Logarithmic / power-of-two** formats space levels geometrically. They match the
  heavy-tailed magnitude distribution and turn multiplications into bit-shifts (cheap
  hardware), but the coarse spacing at large magnitudes costs accuracy. Used in some
  specialized low-power accelerators.
- **Codebook / vector-quantized** formats (AQLM, QuIP# E8 lattice) store a learned
  codebook and represent weights (or groups) by indices. These approach the optimum
  at 2 bits but require a lookup/decode step, trading kernel simplicity for
  accuracy-per-bit.

The tradeoff, restated at the format level: uniform integer formats are
hardware-cheap (a scale and a shift); non-uniform formats are more accurate per bit
but need a lookup table or companding function in the decode path. NF4 is the sweet
spot that reached production; codebooks are the accuracy champions that remain
kernel-limited.

## Accuracy by bit-width across model classes

![Accuracy degradation by bit-width across model classes](../assets/charts/04_accuracy_by_bitwidth.png)

The chart shows how quality degrades as bit-width falls, for three model classes
(illustrative, reflecting typical behavior; the LLM curve is weight-only per-group,
which is why 4-bit is strong, while the CNN curve is weight+activation). Three
observations. First, **all classes are near-lossless at 8 and 6 bits** — the green
safe zone. Second, **at 4 bits the classes diverge**: weight-only LLM quantization
holds up well (~97% retention), CNN weight+activation quantization degrades more
(~96% but with more variance), and diffusion models — which compound error over dozens
of denoising steps — degrade sharply (~88%). Third, **below 4 bits every class falls
off a cliff**, with diffusion the most fragile and LLMs (with advanced methods) the
most resilient. The per-class differences are exactly the sensitivity story of
Section 03: diffusion's iterative compounding and CNNs' activation quantization make
them harder than weight-only LLM compression at the same nominal bits.

This class-dependence is why "what bit-width is safe" has no single answer — it
depends on the model family, whether activations are quantized, and the granularity.
A responsible spec always states all three.

## Mixed precision as a format strategy

Real deployments rarely use one format throughout. **Mixed-precision** models assign
formats per layer or per tensor: 8-bit embeddings and LM head, 4-bit middle-layer
weights, FP16 normalization and softmax, INT32 accumulation. The GGUF k-quants
formalize this within a single file — a "Q4_K_M" model mixes 4-bit, 5-bit, and 6-bit
blocks according to a fixed importance heuristic. Mixed precision is the practical
reconciliation of the format tradeoffs: use the cheapest format each part of the
network can tolerate. Its viability depends on hardware that can execute multiple
formats efficiently and a compiler that can schedule the transitions — the co-design
concern of Section 06.

## Choosing a format by workload

Synthesizing the above into practical guidance:

| Workload | Recommended format(s) | Rationale |
|---|---|---|
| Vision/audio CNN, compute-bound | W8A8 INT8 (per-channel) | Near-lossless, universal HW, compute speedup |
| On-device LLM decode, memory-bound | W4A16 INT4 per-group (or NF4) | 4× weight memory cut = 4× token rate |
| LLM prefill / compute-bound phase | FP8 or W8A8 | Compute speedup; FP8 handles activation outliers |
| Transformer activation quantization | FP8 E4M3 or SmoothQuant+INT8 | Outlier-robust range |
| Sub-4-bit LLM weights | MXFP4 / codebook (QuIP#/AQLM) | Best accuracy-per-bit at the frontier |
| Always-on tinyML sensing | INT8, INT4, or binary | Extreme energy budget; easy tasks |
| Diffusion on edge | INT8 (W8A8), sub-8-bit research | Iterative error compounding limits aggression |
| Low-precision training | BF16 → FP8 → FP4 (with stochastic rounding) | Gradient range preservation |

## Hardware support maturity by format

A format is only as useful as the silicon that runs it. The matrix below summarizes
where each format stands in *edge* hardware specifically (data-center support is
generally one generation ahead).

| Format | Edge silicon support | Edge software maturity | Overall |
|---|---|---|---|
| INT8 | Universal | Universal | 🟢 production |
| INT16 / W8A16 | Wide | Good | 🟢 production |
| INT4 (weight-only) | Flagship + many mid-range | Strong (GGUF/GPTQ/AWQ) | 🟢 production |
| FP16 | Universal | Universal | 🟢 production |
| BF16 | Wide | Good | 🟢 production (mostly training) |
| FP8 | Flagship 2025 NPUs | Emerging | 🟡 emerging |
| INT2 | Newest flagship only | Sparse | 🟡 silicon ahead of SW |
| FP4 / MXFP4 | Newest flagship / DC | Immature | 🟡 frontier |
| NF4 | Runs on any (SW format) | Strong (bitsandbytes) | 🟢 (as SW format) |
| Ternary/binary | Rare native support | Research | 🔴 research |

The consistent pattern — hardware support preceding software maturity by one or more
product cycles for every sub-8-bit format — is the single most important thing to
understand about the format landscape, and it is why every FP4/INT2 claim in this
database carries a confidence flag until independent low-bit accuracy benchmarks
appear.

## Summary

Numeric formats divide into uniform integer types (cheap hardware, uniform
resolution, range-limited) and floating-point types (range-adaptive, outlier-robust,
more complex), with non-uniform (NF4, codebook) and block-floating-point (MX)
families bridging them. INT8 and weight-only INT4 are the production baselines; FP8
is the emerging second baseline; FP4/MXFP4 and INT2 are the hardware-ahead-of-software
frontier; and the microscaling standard is the likely convergence point because it
merges per-group quantization with a hardware-native format. The choice of format is
increasingly made *by the target silicon*, which constrains the techniques of Section
03 — and the LLM-specific methods of Section 05, which we turn to next, are largely
about extracting maximum accuracy from these formats at 4 bits and below.

---

*Next: [05 — LLM-Specific Quantization Methods](./05-llm-methods.md).*
