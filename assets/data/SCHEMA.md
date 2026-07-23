# Master database schema (`master_database.csv`)

The consolidated entity table that Section 16 renders and that Sections 08–13
feed incrementally. One row per distinct entity. Deduplicated in Section 16.

| Column | Meaning | Allowed / example values |
|---|---|---|
| `entity` | Canonical name of the org, product, framework, method, lab, or startup | `Qualcomm`, `GPTQ`, `MIT HAN Lab`, `Neural Magic` |
| `category` | Top-level bucket | `chipmaker`, `framework`, `method`, `research-lab`, `startup`, `benchmark`, `standard` |
| `subcategory` | Finer role | `mobile-soc`, `edge-npu`, `ptq-tool`, `llm-quant`, `compiler`, `academic`, `inference-runtime` |
| `technique_focus` | What quantization angle this entity centers on | free text, e.g. `weight-only INT4 PTQ`, `activation outlier smoothing` |
| `precision_support` | Precisions shipped/supported (hardware) or targeted (software/method) | e.g. `INT8, INT4, FP16`; `W4A16`; `binary/ternary` |
| `maturity` | Deployment reality | `production-shipped`, `sdk-limited`, `research-only` |
| `key_differentiator` | One-line distinguishing claim | free text |
| `notable_output` | Flagship product / paper / release | free text |
| `source_confidence` | Evidence basis | `official-spec`, `sdk-docs`, `paper`, `press`, `inferred`, `speculative` |
| `section_ref` | Section(s) where the entity is discussed | e.g. `09`, `11`, `05;16` |

## Confidence taxonomy (see Section 17 for the full methodology note)

- **official-spec** — vendor datasheet, published spec sheet, or peer-reviewed paper with the exact claim.
- **sdk-docs** — inferred from SDK/compiler documentation or release notes (capability exists in tooling; adoption unverified).
- **paper** — claim from an academic paper; may not be production-adopted.
- **press** — announcement, keynote, or press release without an independent third-party benchmark.
- **inferred** — deduced from adjacent evidence (e.g., ISA support implies a precision) but not explicitly stated.
- **speculative** — roadmap or unreleased-hardware expectation; flagged as low-confidence.

## Maturity taxonomy

- **production-shipped** — used in shipping products at scale (verifiable in released silicon, apps, or widely-used model releases).
- **sdk-limited** — available in an SDK/toolkit or research codebase but with limited or unverified production adoption.
- **research-only** — demonstrated in papers/prototypes; not packaged for general deployment.
