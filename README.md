# Compost NCQ TOPSIS ML data and code

This repository contains only the processed datasets and source code for the nutrient-carbon-quality TOPSIS machine-learning composting study.

Generated images, manuscript files, model checkpoints, SHAP arrays, and exported result tables are intentionally not tracked. They can be regenerated from the included data and code.

## Contents

```text
data/
  __16/ and __18/                 Processed endpoint datasets

src/
  common/                         Shared configuration, modelling pipeline,
                                  optimization helpers, and plotting style
  figures/                        Figure and table-generation scripts
  run_model_pipeline.py            Re-trains endpoint models with checkpointing
  build_result_summaries.py        Aggregates per-endpoint outputs for figures

reproduce_figures.py               Runs summary-building and figure scripts
requirements.txt                   Python package requirements
```

## Endpoints

- NH3-N loss
- N2O-N loss
- CH4-C loss
- CO2-C loss
- TC loss
- TN loss
- Final germination index (GI)

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Train models and generate intermediate outputs:

```bash
python src/run_model_pipeline.py --time-budget 120
```

Build aggregate result tables and regenerate figures/tables:

```bash
python reproduce_figures.py
```

Generated files are written under `outputs/` or inside the corresponding `src/figures/` folders and are ignored by Git.

## Notes

The `data/` files are processed endpoint tables harmonized for this secondary analysis. Please cite the manuscript and the original source dataset referenced there when reusing the data.

## License

Code is released under the MIT License. Processed data are provided for research reuse with attribution to this repository and to the original source dataset cited in the manuscript.
