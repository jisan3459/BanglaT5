# BanglaT5 QA —  Clean MVC Project

This is the Google-Drive-ready project for the BanglaT5-small Full Fine-Tuning vs LoRA study.

## Fixes in this version

1. **Fixed dependency installation in notebooks.** The older notebooks used a `%pip -r "{}...".format(...)` pattern that could fail to resolve the requirements path correctly. This version uses `subprocess.check_call([...])` with the exact Drive path.
2. **Removed the `codecarbon/` project folder name.** That name can shadow the installed Python package `codecarbon` after the project root is added to `sys.path`. Logs now go to **`carbon_logs/`**.
3. **Preprocessing is independent of training dependencies.** `PreprocessingController` no longer imports the training model, CodeCarbon, or PEFT. Notebook 00 can run without CodeCarbon.
4. **Removed `OutputMethod` dependency.** Current CodeCarbon supports CSV output directly; the project imports only `EmissionsTracker`, writes `emissions.csv`, and reads the final energy/emissions values from that CSV.
5. **Added `accelerate` to training requirements**, which Hugging Face Trainer requires.
6. **Added explicit dependency/GPU checks** before training and a CodeCarbon path check to catch any future package-shadowing issue immediately.
7. **Made preprocessing artifact saving idempotent** so rerunning notebook 00 cleanly replaces tokenized/prepared outputs rather than mixing old and new artifacts.
8. Preserved the final research methodology and evaluation logic.

## Important replacement step

If you previously uploaded the old `BanglaT5_Project` folder, **rename or delete that old folder first**, then extract this fresh package so that the new folder is exactly:

```text
MyDrive/BanglaT5_Project/
```

Do not merge this package into the old folder, because the old `codecarbon/` directory may remain and shadow the installed package.

## Folder structure

```text
BanglaT5_Project/
├── data/
│   ├── raw/                 # the 3 targeted CSV files are included
│   └── prepared/            # created by notebook 00
├── models/                  # created automatically
├── carbon_logs/             # CodeCarbon CSV logs; safe name
├── results/
├── requirements/
│   ├── preprocessing.txt
│   ├── training.txt
│   ├── lora.txt
│   └── analysis.txt
├── notebooks/
│   ├── 00_Data_Preprocessing.ipynb
│   ├── 01_Full_Fine_Tuning.ipynb
│   ├── 02_LoRA_r4.ipynb
│   ├── 03_LoRA_r8.ipynb
│   ├── 04_LoRA_r16.ipynb
│   ├── 05_LoRA_r32.ipynb
│   └── 06_Final_Analysis.ipynb
└── src/
    ├── config.py
    ├── data_model.py
    ├── experiment_model.py
    ├── controllers.py
    ├── views.py
    └── analysis.py
```

## MVC mapping

- **Model layer** — `data_model.py` and `experiment_model.py`
- **View layer** — `views.py`
- **Controller layer** — `controllers.py`
- **Configuration** — `config.py`
- **Post-experiment analysis** — `analysis.py`

## Run order

1. `00_Data_Preprocessing.ipynb` — once.
2. Start a fresh T4 GPU runtime and run `01_Full_Fine_Tuning.ipynb`.
3. Use a fresh T4 runtime for each LoRA notebook: r=4, r=8, r=16, r=32.
4. Run `06_Final_Analysis.ipynb` after all five experiments.

## Final methodology preserved

- Base model: `csebuetnlp/banglat5_small`
- Input format: `প্রসঙ্গ: ...
প্রশ্ন: ...
উত্তর:`
- Source length: 768
- Target length: 96
- FP32
- 3 epochs
- Batch size 2
- Gradient accumulation 4 (effective batch 8)
- Seed/data seed 42
- Full FT LR: 5e-5
- LoRA LR: 1e-3
- LoRA ranks: 4, 8, 16, 32
- LoRA alpha: 2 × rank
- Target modules: `q`, `v`
- LoRA dropout: 0.1
- CodeCarbon tracks training only
- Greedy generation (`num_beams=1`, `do_sample=False`)
- Same normalized multi-reference EM and token-level F1
- Blank predictions remain model errors

## Previous-run reference values

The `results/previous_run_reference.csv` file contains the prior final measurements for comparison. Exact energy, CO2, runtime, and potentially small score differences cannot be guaranteed across fresh Colab sessions because the runtime, package builds, and GPU execution can vary.
