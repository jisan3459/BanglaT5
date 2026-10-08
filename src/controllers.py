from .config import ExperimentConfig, ProjectConfig
from .data_model import BanglaRQADataModel, PreparedDataRepository
from .views import ConsoleView


class PreprocessingController:
    """Coordinates the one-time preprocessing pipeline."""

    def __init__(self, config: ProjectConfig, view=ConsoleView):
        self.config = config
        self.view = view

    def run(self):
        data_model = BanglaRQADataModel(self.config)
        processed = data_model.process()
        tokenized = data_model.tokenize(processed)
        data_model.save_artifacts(processed, tokenized)
        self.view.preprocessing_summary(
            processed,
            tokenized,
            self.config.max_source_length,
            self.config.max_target_length,
        )
        return processed, tokenized


class TrainingController:
    """Coordinates one isolated Full-FT or LoRA experiment."""

    def __init__(self, project: ProjectConfig, experiment: ExperimentConfig, view=ConsoleView):
        self.project = project
        self.experiment = experiment
        self.view = view

    def run(self):
        # Lazy import keeps preprocessing independent of CodeCarbon/PEFT.
        from .experiment_model import BanglaT5ExperimentModel

        tokenizer, processed, tokenized = PreparedDataRepository(self.project).load()
        model = BanglaT5ExperimentModel(
            self.project,
            self.experiment,
            tokenizer,
            processed,
            tokenized,
        )
        model.build()

        self.view.experiment_setup(
            self.experiment,
            model.total_params,
            model.trainable_params,
            next(model.model.parameters()).dtype,
        )

        pre = model.sanity_check()
        self.view.health_check("PRE-TRAINING HEALTH CHECK", pre)
        if not pre["loss_finite"] or pre["nan_logits"] or pre["inf_logits"]:
            raise RuntimeError("Pre-training health check failed.")

        train_info = model.train()
        self.view.training_summary(self.experiment.method, train_info)

        post = model.post_training_check(pre["batch"])
        self.view.health_check("POST-TRAINING HEALTH CHECK", post)
        if not post["loss_finite"] or post["nan_logits"] or post["inf_logits"]:
            raise RuntimeError("Post-training health check failed.")

        validation_eval = model.evaluate("validation")
        self.view.evaluation_summary(self.experiment.method, "validation", validation_eval)

        test_eval = model.evaluate("test")
        self.view.evaluation_summary(self.experiment.method, "test", test_eval)

        model_dir = model.save_model()
        summary = model.save_results(train_info, post, validation_eval, test_eval)

        print(f"\nModel saved to: {model_dir}")
        print(
            "Results saved to:",
            self.project.results_dir / f"{self.experiment.run_name}_summary.csv",
        )
        return summary
