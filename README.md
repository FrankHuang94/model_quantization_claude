# Model Quantization for Edge AI — Technical Reference Database

A structured, multi-file technical reference on **model quantization for edge AI**:
core techniques and numerics, LLM-specific methods, hardware/software co-design,
per-vendor silicon roadmaps, the academic research landscape, the startup
ecosystem, standards/benchmarks, and a consolidated competitive database.

> **Generated:** 2026-07-23 · **Scope:** quantization techniques, edge/mobile
> NPU silicon, compiler toolchains, LLM compression, and the research + startup
> ecosystem around them. · **Target length:** 100,000+ words across all sections.

## How to read this repository

- Each section is a standalone file under [`/sections`](./sections). Sections are
  numbered `01`–`17` and are intended to be read in order, but each is
  self-contained enough to consult directly.
- **Charts** are generated with Python (matplotlib/pandas) from data in
  [`/assets/data`](./assets/data) and rendered to PNG in
  [`/assets/charts`](./assets/charts). Every chart is regeneratable from its
  backing CSV/JSON via the scripts in [`/assets/scripts`](./assets/scripts).
- **Diagrams** use Mermaid, which GitHub renders natively in Markdown.
- The **master competitive database** lives at
  [`/assets/data/master_database.csv`](./assets/data/master_database.csv) with
  its schema in [`SCHEMA.md`](./assets/data/SCHEMA.md); Section 16 renders and
  analyzes it.

### Conventions used throughout

Every hardware/technique claim is tagged with a **maturity** level and, for
forward-looking or unreleased items, a **confidence** level:

| Tag | Meaning |
|---|---|
| 🟢 **production-shipped** | Deployed at scale in released silicon, apps, or widely-used model releases. |
| 🟡 **sdk-limited** | Present in an SDK / toolkit / research codebase; limited or unverified production adoption. |
| 🔴 **research-only** | Demonstrated in papers or prototypes; not packaged for general deployment. |
| ⚠️ **confidence flag** | Roadmap / unreleased-hardware / vendor-claim not independently verified. |

Vendor performance claims (e.g. "X% faster", "N TOPS") are attributed to the
vendor and flagged where no independent third-party benchmark corroborates them.

## Table of contents

| # | Section | Focus |
|---|---|---|
| 01 | [Executive summary](./sections/01-exec-summary.md) | Cross-cutting view of the accuracy/efficiency frontier and who leads in silicon vs. tooling |
| 02 | [Development history (2015–present)](./sections/02-history.md) | From BinaryConnect/XNOR-Net through INT8 PTQ to the LLM-quantization era |
| 03 | [Core techniques and theory](./sections/03-core-techniques.md) | PTQ vs QAT, granularity, calibration, outlier handling |
| 04 | [Precision formats and numerics](./sections/04-precision-formats.md) | INT8/4/2, FP8/FP4, block FP, log/non-uniform, binary/ternary |
| 05 | [LLM-specific quantization methods](./sections/05-llm-methods.md) | GPTQ, AWQ, SmoothQuant, SpQR, QuIP#, HQQ, GGUF, QLoRA, KV-cache |
| 06 | [Hardware–software co-design](./sections/06-hw-sw-codesign.md) | NPU/DSP matrix engines, compilers, sparsity+quant, bandwidth |
| 07 | [Pros, cons, and deployment tradeoffs](./sections/07-tradeoffs.md) | Accuracy loss, latency/energy gains, failure modes |
| 08 | [Apple roadmap](./sections/08-apple.md) | Neural Engine, Core ML, MLX, Apple Intelligence |
| 09 | [Qualcomm roadmap](./sections/09-qualcomm.md) | Hexagon NPU, AIMET, Snapdragon, AI Engine Direct |
| 10 | [MediaTek roadmap](./sections/10-mediatek.md) | APU, NeuroPilot, Dimensity |
| 11 | [Other major players](./sections/11-other-players.md) | Samsung, Google, NVIDIA, Intel, AMD, ARM, Huawei |
| 12 | [Academic research landscape](./sections/12-academia.md) | Leading labs, landmark papers, active directions |
| 13 | [Startup landscape](./sections/13-startups.md) | Compression/edge-inference startups, funding, differentiators |
| 14 | [Future roadmap (3–5 years)](./sections/14-future.md) | Sub-4-bit, extreme quant, multimodal/diffusion, co-design |
| 15 | [Standards and benchmarks](./sections/15-standards-benchmarks.md) | MLPerf, ONNX quant spec, benchmark coverage |
| 16 | [Master competitive & research database](./sections/16-master-database.md) | Consolidated 75–100+ entry table + summary charts |
| 17 | [Appendix: glossary & methodology](./sections/17-appendix.md) | Terms, confidence-level methodology |

## Repository layout

```
.
├── README.md
├── sections/
│   ├── 01-exec-summary.md
│   ├── 02-history.md
│   ├── … (through)
│   └── 17-appendix.md
└── assets/
    ├── charts/     # generated PNG chart images
    ├── data/       # CSV/JSON backing every chart + master_database.csv
    └── scripts/    # Python generators (chartstyle.py + per-section scripts)
```

## Regenerating the charts

```bash
pip install matplotlib pandas numpy
cd assets/scripts
python 01_pipeline_maturity.py   # etc. — one script per section's charts
```

All scripts import `chartstyle.py` for a consistent palette and write into
`../charts/`.

---

*This is a technical reference compiled from public specifications, SDK
documentation, academic papers, and vendor announcements. Where a claim could
not be independently verified it is flagged. Corrections welcome.*
