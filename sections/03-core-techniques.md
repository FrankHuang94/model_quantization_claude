# 03. Core Quantization Techniques and Theory

> **Section scope.** The conceptual and mathematical foundation shared by every
> quantization method: how the quantization mapping works, the axes along which
> schemes differ (symmetric/asymmetric, granularity, weight-only vs. weight+
> activation, static/dynamic, PTQ/QAT), how calibration and rounding are chosen,
> and how outliers are handled. This section is deliberately technique-agnostic;
> the LLM-specific *named* methods (GPTQ, AWQ, etc.) build on these primitives and
> are treated in Section 05, and the numeric formats (INT4, FP8, etc.) in Section 04.

## The quantization mapping

At its core, quantization replaces a high-precision tensor **x** (FP32 or FP16/BF16)
with a low-precision representation that can be reconstructed approximately. The
overwhelmingly dominant scheme is **uniform affine quantization**, defined by a
**scale** `s` (a positive real number) and a **zero-point** `z` (an integer):

```
quantize:    x_q = clip( round(x / s) + z,  q_min,  q_max )
dequantize:  x_hat = s * (x_q - z)
```

The scale `s` sets the spacing between representable values (the *step size*), and
the zero-point `z` is the integer that maps to the real value 0.0 — important
because exact representation of zero matters for padding, ReLU, and sparsity. The
clip to `[q_min, q_max]` (e.g. `[-128, 127]` for signed INT8, `[0, 255]` for
unsigned) is where information is irreversibly lost for any value outside the
representable range. The reconstruction error `x - x_hat` is the **quantization
error**, and essentially all of quantization theory is about minimizing its effect
on the network's output.

Two sources of error coexist. **Rounding error** affects values inside the range
and is bounded by `s/2` per element; it behaves like additive noise roughly uniform
on `[-s/2, s/2]`. **Clipping error** affects values outside `[q_min, q_max]` and is
unbounded in principle; a single large outlier that gets clipped can distort a
result badly. The central tension of calibration — choosing `s` — is that a *large*
`s` (wide range) reduces clipping error but coarsens the step and increases
rounding error, while a *small* `s` (narrow range) does the opposite. Every
calibration method is a different answer to this bias–variance-like tradeoff.

A useful summary metric is the **signal-to-quantization-noise ratio (SQNR)**, the
ratio of signal power to quantization-noise power, usually in dB. For a
well-matched uniform quantizer, SQNR improves by roughly 6 dB per additional bit —
the classic "6 dB/bit" rule from signal processing. This is why each bit removed
roughly quadruples the noise power, and why the accuracy cliff below 4 bits is so
steep: there simply is not enough SQNR left to represent the signal once outliers
consume most of the range.

## Symmetric vs. asymmetric quantization

The first design axis is whether the quantization range is **symmetric** about zero
or **asymmetric** (affine with a non-zero zero-point).

- **Symmetric quantization** forces `z = 0`, so the mapping is `x_q = round(x/s)`
  with the range `[-s·q_max, s·q_max]` centered on zero. It is simpler and faster
  in hardware — the integer matrix multiply needs no zero-point-correction terms —
  and it exactly represents zero for free. It is the natural choice for **weights**,
  which are typically distributed roughly symmetrically around zero, and for
  hardware that wants the cheapest possible integer datapath.
- **Asymmetric quantization** allows `z ≠ 0`, shifting the range to fit the actual
  data. It is the better choice for **activations** that are one-sided — the classic
  example is post-ReLU activations, which are all non-negative; a symmetric quantizer
  would waste half its codes (the negative half) on values that never occur, halving
  the effective precision. Asymmetric quantization uses the full `[0, 255]` range,
  doubling resolution, at the cost of zero-point-correction arithmetic in the matmul.

In practice the common production configuration is **symmetric weights, asymmetric
activations** (or symmetric both, if the hardware or format prefers it). The choice
interacts with hardware: some NPUs support only symmetric, some support both, and
the compiler may silently convert. The zero-point correction for asymmetric matmul
adds terms that must be folded efficiently, and getting this right is a
compiler concern (Section 06).

| Property | Symmetric | Asymmetric (affine) |
|---|---|---|
| Zero-point | Fixed at 0 | Learned/calibrated integer |
| Exact zero | Always | Always (by construction) |
| Range utilization for one-sided data | Poor (wastes half) | Full |
| Matmul overhead | None | Zero-point correction terms |
| Typical use | Weights | Activations (esp. post-ReLU) |
| Hardware support | Universal | Common but not universal |

## Granularity: per-tensor, per-channel, per-group

The second — and arguably most consequential — axis is **granularity**: how many
distinct `(s, z)` pairs are used per tensor. Finer granularity fits the data better
(less error) but stores more scale metadata and complicates the hardware.

- **Per-tensor** (a single `s, z` for the whole weight or activation tensor) is the
  cheapest and the coarsest. It fails when different channels have very different
  magnitude ranges — common in depthwise convolutions and transformer weights —
  because one scale must accommodate the widest channel, crushing precision for the
  rest.
- **Per-channel** (a separate scale per output channel of a weight tensor) is the
  production standard for weights. It costs one scale per channel (negligible
  storage) and dramatically improves accuracy because each channel gets a range
  matched to its own distribution. Activations are usually *not* quantized
  per-channel because the channel dimension is the contraction (reduction) dimension
  of the matmul, and per-channel activation scales do not factor cleanly out of the
  dot product; per-*token* activation scaling is used instead where dynamic ranges
  vary by token.
- **Per-group / per-block** (a separate scale for each contiguous group of, say, 32,
  64, or 128 weights within a channel) is the granularity that made 4-bit and
  sub-4-bit LLM quantization work. A group size of 128 is the common default; 32 or
  64 gives more accuracy at more overhead. Per-group quantization is essentially
  **block floating point** applied to integers, and it is the conceptual bridge to
  the microscaling (MXFP) hardware formats of Section 04 — those formats standardize
  per-group scaling *into the numeric type itself*.

![Quantization granularity vs. accuracy retention](../assets/charts/03_granularity_accuracy.png)

The chart above (illustrative, reflecting typical weight-only LLM PTQ behavior)
shows the payoff. At 8 bits, granularity barely matters — everything is
near-lossless. As bits fall, the curves fan out dramatically: at 3–4 bits, moving
from per-tensor to per-group (g=128) can be the difference between a broken model
and a usable one, and per-group (g=32) extends usability to lower bits still. The
cost is **scale overhead**: a group size of 128 with a 16-bit scale adds `16/128 =
0.125` bits per weight; a group size of 32 adds `0.5` bits per weight — non-trivial
at a nominal 4-bit budget, which is why group size is a real accuracy/size knob, not
a free lunch. Sub-4-bit methods therefore also quantize the *scales* (double
quantization, as in QLoRA) to claw back the overhead.

| Granularity | Scales per tensor | Accuracy | Overhead | Typical use |
|---|---|---|---|---|
| Per-tensor | 1 | Lowest | Minimal | Legacy / simple INT8 activations |
| Per-channel | # output channels | Good | Negligible | Standard for weights (all bit-widths) |
| Per-group (g=128) | # channels × (in/128) | Very good | ~0.125 bits/weight | 4-bit LLM weights (default) |
| Per-group (g=32) | # channels × (in/32) | Best | ~0.5 bits/weight | Sub-4-bit / accuracy-critical |
| Per-token (activations) | # tokens | Good for activations | Dynamic, cheap | Dynamic activation quantization |

## Weight-only vs. weight-and-activation quantization

Whether to quantize only the weights, or the activations too, is the axis that most
determines *what kind of speedup you get*, and it is where the vision and LLM worlds
diverge most sharply.

- **Weight-only quantization** (e.g. W4A16: 4-bit weights, 16-bit activations)
  compresses the weights but keeps activations and the matmul accumulation in
  higher precision. The matmul is performed by dequantizing weights on the fly (or
  using specialized mixed-precision kernels). The benefit is almost entirely
  **memory**: less weight storage and less weight traffic from DRAM. For LLM
  decoding — which is memory-bandwidth-bound, streaming all weights per token —
  this directly buys ~4× faster decoding, which is why W4A16 dominates on-device
  generative AI. It does *not* speed up compute-bound workloads, because the actual
  arithmetic still happens at high precision.
- **Weight-and-activation quantization** (e.g. W8A8: both 8-bit) quantizes both
  operands so the matmul runs on integer (or low-FP) tensor units at full low-
  precision throughput. This is the path to **compute** speedups and is standard for
  vision/audio CNNs on NPUs, where the workload is compute-bound. The catch is the
  activation outlier problem (Section 02): quantizing activations to low precision
  is much harder than quantizing weights, which is why W8A8 needs SmoothQuant-style
  preprocessing and why W4A4 remains research-stage.

The rule of thumb: **weight-only for memory-bound (LLM decoding), weight+activation
for compute-bound (vision, prefill).** Many real systems are hybrids — an LLM might
use W4A16 for decoding but benefit from W8A8 or FP8 during the compute-bound prefill
of a long prompt.

## Static vs. dynamic quantization

For activations specifically, the quantization parameters can be fixed ahead of time
or computed at runtime.

- **Static quantization** pre-computes activation scales/zero-points offline from a
  calibration dataset and bakes them into the model. Runtime is cheapest (no range
  computation), and the whole graph can run in integer end-to-end, which is what
  NPUs want. The risk is *calibration mismatch*: if runtime activations differ from
  the calibration data's distribution, the fixed range clips or wastes precision.
- **Dynamic quantization** computes activation ranges on the fly from the actual
  tensor being processed (per inference, often per-token). It adapts perfectly to
  the data — no calibration set needed — at the cost of a runtime reduction to find
  min/max. Dynamic **per-token** activation quantization is common in LLM serving
  because it handles the varying dynamic range of different tokens gracefully and
  needs no calibration.

Weights are always effectively "static" (known at deploy time). The static/dynamic
choice is about activations, and it trades runtime overhead against robustness to
distribution shift. NPUs generally favor static (for the fully-integer pipeline);
flexible CPU/GPU LLM runtimes often use dynamic per-token.

## PTQ vs. QAT: the central methodological fork

The single biggest decision in a quantization project is **post-training
quantization (PTQ)** versus **quantization-aware training (QAT)**.

**PTQ** takes an already-trained FP model and quantizes it without further training,
using at most a small calibration set (a few hundred samples) to set activation
ranges and, in advanced variants, to locally optimize weights (AdaRound, GPTQ).
PTQ is fast (minutes to a few GPU-hours), needs little or no data, requires no
training pipeline or labels, and is the default first attempt. It works excellently
at INT8 and, with good methods and per-group granularity, at 4-bit for LLM weights.

**QAT** inserts simulated quantization ("fake-quant" nodes that round in the forward
pass) into the model and *continues training* (or fine-tunes), so the weights adapt
to the quantization. Gradients flow through the non-differentiable rounding via the
**straight-through estimator** (STE), which treats `round()` as the identity for the
backward pass. QAT recovers substantially more accuracy than PTQ, especially at low
bit-widths, because the network learns weights that are robust to being quantized.
The cost is real: a full or partial training run, the training data and labels, the
compute, and the engineering of a QAT pipeline.

![PTQ vs. QAT accuracy by bit-width](../assets/charts/03_ptq_vs_qat.png)

The chart makes the tradeoff concrete (illustrative values). At INT8 the gap is
negligible — use PTQ. As bit-width falls, the gap widens: at INT4 QAT recovers
several points PTQ leaves on the table, and at INT2 QAT is often the difference
between a functioning and a broken model. This yields a simple heuristic: **PTQ
until it stops being good enough, then QAT** — and "good enough" arrives sooner the
lower the bit-width and the harder the task.

A modern middle ground is **parameter-efficient QAT** and **quantization-native
training**. QLoRA-style approaches fine-tune low-rank adapters on top of frozen
quantized weights, capturing much of QAT's benefit at a fraction of the cost.
BitNet-style quantization-native training (Section 02, 04) goes further, training
the model in low precision from scratch so no post-hoc quantization is needed at all.

## The PTQ-vs-QAT (and technique-selection) decision tree

The following decision tree encodes the practical selection logic. It is
deliberately opinionated toward the cheapest technique that meets the accuracy bar,
escalating only when necessary.

```mermaid
flowchart TD
    START([Model to deploy]) --> Q1{Target bit-width?}
    Q1 -- INT8/W8A8 --> Q2{Vision/CNN or transformer?}
    Q2 -- CNN --> P1[Per-channel weights +<br/>static asym activations, PTQ]
    Q2 -- Transformer --> Q3{Activations quantized?}
    Q3 -- No, W8A16 --> P2[Per-channel PTQ<br/>weight-only]
    Q3 -- Yes, W8A8 --> P3[SmoothQuant + per-token<br/>dynamic activations, PTQ]
    P1 --> ACC{Accuracy OK?}
    P2 --> ACC
    P3 --> ACC
    Q1 -- 4-bit weights --> Q4{Memory- or compute-bound?}
    Q4 -- Memory LLM decode --> P4[Weight-only W4A16<br/>GPTQ/AWQ, per-group g=128]
    Q4 -- Compute-bound --> P5[Consider FP8 or W4A8<br/>hardware-dependent]
    P4 --> ACC
    P5 --> ACC
    Q1 -- Sub-4-bit --> Q5{Can you retrain?}
    Q5 -- No --> P6[Rotation/codebook PTQ<br/>QuIP#/AQLM, g=32<br/>research-stage]
    Q5 -- Yes --> P7[QAT or quantization-native<br/>BitNet-style training]
    P6 --> ACC
    P7 --> ACC
    ACC -- Yes --> SHIP([Compile & deploy])
    ACC -- No --> ESC{Escalation options}
    ESC --> E1[Finer granularity g=128 to g=32]
    ESC --> E2[Protect sensitive layers<br/>higher precision]
    ESC --> E3[Add outlier handling<br/>SmoothQuant/rotation]
    ESC --> E4[Escalate PTQ to QAT]
    E1 --> ACC
    E2 --> ACC
    E3 --> ACC
    E4 --> ACC
```

The escalation loop at the bottom is the real work of a quantization engineer:
finer granularity, mixed precision for sensitive layers (embeddings, the final
projection, the first and last layers are common culprits), outlier handling, and
finally QAT. Each step costs more effort and buys more accuracy.

## Calibration methods

Calibration is the process of choosing the quantization range — the clipping
thresholds — for weights and especially activations, from a small representative
dataset. The method matters more than beginners expect, because the outlier-vs-
resolution tradeoff is decided here.

- **Min–max** uses the observed minimum and maximum as the range. Simple and
  outlier-sensitive: a single extreme activation sets the range and coarsens
  everything else. Fine for weights, often too loose for activations.
- **Percentile / clipping** discards the extreme tails (e.g. clip at the 99.9th
  percentile), trading a little clipping error for much better resolution. A common,
  robust default for activations.
- **MSE (mean-squared-error) minimization** searches for the clipping threshold that
  minimizes the reconstruction error of the tensor, balancing rounding and clipping
  error analytically or by grid search. Strong general-purpose choice.
- **KL-divergence / entropy calibration** (TensorRT's original method) chooses the
  threshold that minimizes the KL divergence between the FP and quantized activation
  distributions — i.e. preserves the *information* in the distribution rather than
  its extremes. Effective for activation-heavy vision models.
- **Cross-layer equalization + bias correction** (from Data-Free Quantization)
  rescales consecutive layers to even out channel ranges and corrects the mean shift
  quantization introduces, sometimes eliminating the need for calibration data
  entirely.
- **Learned / reconstruction-based** (AdaRound, BRECQ) go beyond range selection to
  optimize the rounding of each weight to minimize the layer or block output error,
  using the calibration set as a tiny optimization objective. This blurs into the
  GPTQ family of Section 05.

| Calibration method | Outlier robustness | Data needed | Cost | Typical use |
|---|---|---|---|---|
| Min–max | Low | Tiny | Trivial | Weights; quick baselines |
| Percentile clipping | Medium–High | Small | Low | Activations (robust default) |
| MSE minimization | High | Small | Low–Medium | General-purpose PTQ |
| KL / entropy | High | Small | Medium | Vision activations (TensorRT) |
| Cross-layer equalization | High | None–tiny | Low | Data-free INT8 |
| AdaRound / reconstruction | High | Small | Medium–High | INT4 PTQ; precursor to GPTQ |

The choice of **calibration data** is as important as the method: it must match the
deployment distribution. A model calibrated on clean images and deployed on noisy
ones, or an LLM calibrated on web text and used for code, will have mis-set
activation ranges. This is a frequent, under-appreciated source of quantization
accuracy loss, and it is why dynamic activation quantization (no calibration) is
attractive when the deployment distribution is unpredictable.

## Rounding: nearest is not optimal

The naive assumption is that each weight should be rounded to its nearest
quantization level. **AdaRound** (Section 02) showed this is provably suboptimal:
the goal is to minimize the impact on the *layer's output*, not on the individual
weights, and because weights interact through the matmul, a coordinated choice of
rounding directions (some up, some down) can cancel errors and preserve the output
far better than independent nearest-rounding. This "learned rounding" insight is the
theoretical seed of GPTQ, which scales the idea to billions of parameters by
processing weights in a Hessian-informed order and compensating each quantization
step's error in the yet-to-be-quantized weights. The takeaway for this section:
**rounding is a degree of freedom, not a fixed rule**, and exploiting it is a large
part of what separates advanced PTQ from naive PTQ.

## Outlier handling: the make-or-break problem

As established in Section 02, activation outliers are the central obstacle to
low-bit and to activation quantization. The core techniques, in increasing
sophistication:

1. **Clipping** the outliers (percentile calibration) — cheap, lossy, a first line
   of defense.
2. **Mixed-precision decomposition** — keep the outlier dimensions/weights in higher
   precision (FP16), the rest low (LLM.int8(), SpQR). Effective but needs
   mixed-precision execution.
3. **Difficulty migration** — SmoothQuant moves activation outliers into weights via
   an offline per-channel rescale that preserves the product; AWQ scales to protect
   salient weights. Training-free, hardware-friendly.
4. **Rotation / incoherence** — QuIP#, QuaRot, SpinQuant multiply by orthogonal
   matrices that spread outlier energy across all dimensions, producing smooth,
   easily-quantized distributions, with the rotations fused away at inference.
5. **Prevention** — quantization-native training and architectural changes that
   suppress outlier formation in the first place.

The general trajectory — clip, isolate, migrate, rotate, prevent — mirrors the
historical arc of the field and is the single most useful mental model for
understanding why any given method exists.

## Error propagation and sensitivity

Not all layers are equally sensitive to quantization, and understanding *why* guides
mixed-precision allocation. Layers with high **Hessian curvature** (where small
weight perturbations cause large loss changes) are sensitive and deserve higher
precision; HAWQ and related methods formalize this. Empirically, the usual sensitive
suspects are: **input embeddings and the output/LM-head projection** (large,
directly touch the vocabulary), **the first and last layers** (they interface with
raw inputs/outputs), **normalization and attention-softmax paths** (dynamic-range
sensitive), and **any layer feeding a residual stream carrying outliers**. A common
production pattern is "quantize everything to 4-bit except keep the embedding and LM
head at 8-bit," which costs little size but recovers meaningful accuracy. The
general principle: **spend your precision budget where the Hessian says it matters.**

## Putting the axes together: the technique-tradeoff table

The axes above are orthogonal knobs, and a concrete deployment picks a value on
each. The table condenses the tradeoffs across the primary technique families,
using the maturity taxonomy.

| Technique family | Accuracy retention | Compute overhead (to apply) | Tooling maturity | Best fit |
|---|---|---|---|---|
| Per-tensor INT8 PTQ | Medium | Trivial | 🟢 High | Legacy / simple pipelines |
| Per-channel INT8 PTQ | High | Trivial | 🟢 High | Vision/audio default |
| INT8 QAT | Very high | High (retraining) | 🟢 High | Accuracy-critical low-bit |
| W8A8 + SmoothQuant | High | Low (offline rescale) | 🟡 Medium | Server LLM compute speedup |
| Weight-only W4A16 (GPTQ/AWQ, g=128) | High | Low–Medium (calibration) | 🟢 High | On-device LLM decode |
| W4A16 + protected layers | High | Low–Medium | 🟢 High | LLM accuracy tuning |
| Rotation W4A4 (QuaRot/SpinQuant) | Medium–High | Medium | 🟡 Low–Medium | Research; compute+memory |
| Codebook 2-bit (QuIP#/AQLM) | Medium | High (optimization) | 🟡 Low | Extreme compression, slow kernels |
| Quantization-native (BitNet) | High (if trained so) | Very high (full training) | 🔴 Research | Native low-bit models |

## Choosing granularity, symmetry, and calibration together

A final integrative point: these choices are not independent, and the interactions
matter. Symmetric per-channel weights with per-group scales and MSE-calibrated
per-token dynamic activations is a coherent, strong configuration for 4-bit LLMs;
mixing incompatible choices (e.g. per-tensor symmetric activations with min-max
calibration on outlier-heavy transformer activations) is a recipe for failure. The
compiler and target hardware constrain the feasible combinations — an NPU that only
supports symmetric per-tensor activations forecloses the per-token dynamic option
regardless of what the algorithm prefers. This is why quantization is a co-design
problem and not a pure-software one, and it is the bridge to Section 06.

## Summary

The theory of quantization reduces to a handful of orthogonal choices — the affine
mapping and its scale/zero-point, symmetric vs. asymmetric, granularity (per-tensor/
channel/group/token), weight-only vs. weight+activation, static vs. dynamic, PTQ vs.
QAT — plus the two cross-cutting concerns of calibration (choosing ranges) and
outlier handling (surviving the values that break naive schemes). Every named method
in Section 05 is a specific, cleverly-engineered point in this design space, and
every hardware capability in Sections 06 and 08–11 either enables or forecloses
particular combinations. The next section drills into the *representations*
themselves — the integer and floating-point numeric formats that these techniques
target.

---

*Next: [04 — Precision Formats and Numerics](./04-precision-formats.md).*
