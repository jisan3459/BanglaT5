# Validation report

The fixed project was rechecked before packaging.

Checks completed:

- All Python source files compile successfully.
- All code cells in all seven notebooks compile successfully.
- The three raw targeted-domain CSVs are present and have the expected row counts: train 3035, validation 831, test 886.
- The raw files are copied without changing their rows or split membership.
- No top-level project directory named `codecarbon/` exists; CodeCarbon logs use `carbon_logs/`.
- `src/data_model.py` has no CodeCarbon or PEFT dependency.
- `PreprocessingController` does not import the training model.
- Training uses a lazy `from codecarbon import EmissionsTracker` import only when training starts.
- The obsolete `OutputMethod` import/argument was removed.
- Notebook package installation uses `subprocess.check_call()` with an exact path rather than a `%pip ... .format(...)` expression.
- `accelerate` is included in training requirements.
- LoRA notebooks remove the unused conflicting `torchao` package before installing PEFT.
- Full FT settings remain: FP32, 3 epochs, LR 5e-5, batch 2, gradient accumulation 4, max grad norm 1.0, seed/data_seed 42.
- LoRA settings remain: ranks 4/8/16/32, alpha 2×rank, LR 1e-3, target modules q/v, dropout 0.1, same FP32/batch/epoch settings.
- Evaluation remains greedy generation with `num_beams=1`, `do_sample=False`, `max_new_tokens=96`, multi-reference max EM/F1, and blank predictions kept as errors.
- CodeCarbon remains wrapped around training only.

Dataset note: the supplied targeted CSVs contain a small number of repeated `question_id` values (train and test), exactly as in the original files used for the previous experiments. They are intentionally preserved; the fixed code does not drop, deduplicate, or alter these rows.

The prior final result values are retained only as a reference in `results/previous_run_reference.csv`. Fresh Colab runtime timing, energy, CO2, and potentially small model-score differences can vary with software/hardware execution conditions.
