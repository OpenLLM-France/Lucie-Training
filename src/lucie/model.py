import lightning.pytorch as pl
import torch
from transformers import AutoModelForCausalLM, get_cosine_schedule_with_warmup
from transformers.modeling_outputs import MaskedLMOutput


class LucieModel(pl.LightningModule):
    """
    Pytorch Lightning model class

    Parameters
    ----------
    model_config: str
        A JSON file in which the configuration of the model is stored
    learning_rate: float, default15e-5
        Learning rate
    weight_decay: float, default=0.1
        Weight decay rate
    warmup_steps: int, default=2_000
        Number of warmup steps
    """

    def __init__(
        self,
        model_config: str = None,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.1,
        warmup_steps: int = 2_000,
        num_cycles: float = 0.1,
    ):
        super().__init__()

        self.model_config = model_config
        self.model = None

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.num_cycles = num_cycles

        self.save_hyperparameters()

    def configure_model(self) -> None:
        if self.model is not None:
            return

        self.model = AutoModelForCausalLM.from_config(self.model_config)

    def forward(self, batch: dict) -> MaskedLMOutput:
        """
        Perform forward pass on batch

        Parameters
        ----------
        batch: dict
            mini-batch

        Returns
        -------
        CausalLMOutput
            Model output
        """

        return self.model(**batch)

    def configure_optimizers(self):
        """
        Prepare optimizer for training

        Returns
        -------
        (list, list): list of optimizers and list of schedulers
            Optimizers and schedulers
        """

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            weight_decay=self.weight_decay,
            lr=self.learning_rate,
            betas=(0.9, 0.95),
        )

        scheduler = {
            "scheduler": get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.warmup_steps,
                num_cycles=self.num_cycles,
            ),
            "interval": "step",
            "name": "lr",
        }

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx) -> float:
        """
        Perform a training step.

        Parameters
        ----------
        batch: dict
            Mini-batch.
        batch_idx: int
            Mini-batch index.

        Returns
        -------
        loss: float
            Model loss computed on the mini-batch
        """

        output = self.model(**batch)
        self.log("train_loss", output.loss, on_epoch=True, on_step=True, sync_dist=True)

        return output.loss
