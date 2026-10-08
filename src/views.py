class ConsoleView:
    """Minimal notebook-facing presentation layer."""

    @staticmethod
    def preprocessing_summary(processed, tokenized, max_source_length, max_target_length):
        print("=" * 70)
        print("PREPROCESSING COMPLETE")
        print("=" * 70)
        for name in ("train", "validation", "test"):
            df = processed[name]
            print(
                f"{name.upper():10s} | rows={len(df):4d} | "
                f"unanswerable={(df['is_answerable'] == 0).sum():3d} | "
                f"missing_targets={df['target_text'].isna().sum():2d}"
            )
        print(f"Max source length: {max_source_length}")
        print(f"Max target length: {max_target_length}")
        print("Max tokenized train source:", max(len(x) for x in tokenized["train"]["input_ids"]))
        print("Max tokenized train target:", max(len(x) for x in tokenized["train"]["labels"]))

    @staticmethod
    def experiment_setup(experiment, total_params, trainable_params, dtype):
        print("=" * 70)
        print(f"{experiment.method.upper()} - SETUP")
        print("=" * 70)
        print("Rank:", experiment.rank if experiment.rank is not None else "N/A")
        print("Learning rate:", experiment.learning_rate)
        print("Epochs:", experiment.num_train_epochs)
        print(
            "Effective batch size:",
            experiment.train_batch_size * experiment.gradient_accumulation_steps,
        )
        print("Model dtype:", dtype)
        print("Total parameters:", f"{total_params:,}")
        print("Trainable parameters:", f"{trainable_params:,}")
        print("Trainable percentage:", f"{100 * trainable_params / total_params:.4f}%")

    @staticmethod
    def health_check(label, check):
        print(f"\n{label}")
        print("Loss:", check["loss"])
        print("Loss finite:", check["loss_finite"])
        print("NaN logits:", check["nan_logits"])
        print("Inf logits:", check["inf_logits"])

    @staticmethod
    def training_summary(method, train_info):
        print("\n" + "=" * 70)
        print(f"{method.upper()} - TRAINING RESULTS")
        print("=" * 70)
        print("Training metrics:", train_info["metrics"])
        print(f"Training time: {train_info['training_time_seconds']:.2f} seconds")
        print(f"Training time: {train_info['training_time_seconds'] / 60:.2f} minutes")
        print(f"Peak GPU memory: {train_info['peak_gpu_memory_gb']:.3f} GB")
        print(f"Energy consumed: {train_info['energy_kwh']:.6f} kWh")
        print(f"CO2 emissions: {train_info['co2_kg']:.8f} kg CO2eq")

    @staticmethod
    def evaluation_summary(method, split, result):
        print("\n" + "=" * 70)
        print(f"{method.upper()} - {split.upper()} RESULTS")
        print("=" * 70)
        print(f"Exact Match (EM): {result['em']:.2f}%")
        print(f"Token-level F1: {result['f1']:.2f}%")
        print(f"Exact matches: {result['exact_matches']} / {len(result['results'])}")
        print(f"Blank predictions: {result['blank_predictions']} / {len(result['results'])}")
        print(f"Blank percentage: {result['blank_percentage']:.2f}%")
