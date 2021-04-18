import os

from kashgari.callbacks import EvalCallBack
from wandb.keras import WandbCallback

import wandb


class F1_WandbCallback(WandbCallback):
    def __init__(
        self,
        kashgari_eval_callback: EvalCallBack,
        monitor="val_loss",
        verbose=0,
        mode="auto",
        save_weights_only=False,
        log_weights=False,
        log_gradients=False,
        save_model=True,
        training_data=None,
        validation_data=None,
        labels=[],
        data_type=None,
        predictions=36,
        generator=None,
        input_type=None,
        output_type=None,
        log_evaluation=False,
        validation_steps=None,
        class_colors=None,
        log_batch_frequency=None,
        log_best_prefix="best_",
        save_graph=True,
    ):
        """
        Integrate wandbcallback and f1-score eval callback

        wandbcallback : TODO link
        evalcallback :  TODO link
        Args:
            kashgari_eval_callback (EvalCallBack): [description]
        """
        super().__init__(
            monitor=monitor,
            verbose=verbose,
            mode=mode,
            save_weights_only=save_weights_only,
            log_weights=log_weights,
            log_gradients=log_gradients,
            save_model=save_model,
            training_data=training_data,
            validation_data=validation_data,
            labels=labels,
            data_type=data_type,
            predictions=predictions,
            generator=generator,
            input_type=input_type,
            output_type=output_type,
            log_evaluation=log_evaluation,
            validation_steps=validation_steps,
            class_colors=class_colors,
            log_batch_frequency=log_batch_frequency,
            log_best_prefix=log_best_prefix,
            save_graph=save_graph,
        )
        self.kashgari_eval_callback = kashgari_eval_callback

    def on_epoch_end(self, epoch, logs={}):
        # Run kashgari callback first
        self.kashgari_eval_callback.on_epoch_end(epoch=epoch)
        # update the log will newest f1 score, precision, recall
        assert (
            len(self.kashgari_eval_callback.logs) > 0
        ), "you must eval the performance before you send to Dashboard"
        logs.update(self.kashgari_eval_callback.logs[-1])
        if self.log_weights:
            wandb.log(self._log_weights(), commit=False)

        if self.log_gradients:
            wandb.log(self._log_gradients(), commit=False)

        if (
            self.input_type
            in (
                "image",
                "images",
                "segmentation_mask",
            )
            or self.output_type in ("image", "images", "segmentation_mask")
        ):
            if self.generator:
                self.validation_data = next(self.generator)
            if self.validation_data is None:
                wandb.termwarn(
                    "No validation_data set, pass a generator to the callback."
                )
            elif self.validation_data and len(self.validation_data) > 0:
                wandb.log(
                    {"examples": self._log_images(num_images=self.predictions)},
                    commit=False,
                )

        wandb.log({"epoch": epoch}, commit=False)
        wandb.log(logs, commit=True)
        self.current = logs.get(self.monitor)
        if self.current and self.monitor_op(self.current, self.best):
            if self.log_best_prefix:
                wandb.run.summary[
                    "%s%s" % (self.log_best_prefix, self.monitor)
                ] = self.current
                wandb.run.summary["%s%s" % (self.log_best_prefix, "epoch")] = epoch
                if self.verbose and not self.save_model:
                    print(
                        "Epoch %05d: %s improved from %0.5f to %0.5f"
                        % (epoch, self.monitor, self.best, self.current)
                    )
            if self.save_model:
                self._save_model(epoch)
                print(
                    "Performance Improved! saving model ...",
                )
            self.best = self.current
