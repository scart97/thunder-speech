import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from thunder.data.datamodule import ManifestDatamodule
from thunder.quartznet.module import QuartznetModule
from callbacks.finetune import FinetuneEncoderDecoder
from callbacks.logging import LogSpectrogramsCallback, LogResultsCallback
from pytorch_lightning.callbacks import LearningRateMonitor
import wandb

pl.utilities.seed.seed_everything(8)

hyperparameter_defaults = dict(bs=10, lr_frozen=8e-4, lr_unfrozen=8e-5, betas=[0.8, 0.5])

wandb.init(config=hyperparameter_defaults, project="thunder-speech", entity="madeupmasters")
config = wandb.config
full_dm = ManifestDatamodule(
    train_manifest="train/data/manifests/prepared_train_manifest.json",
    val_manifest="train/data/manifests/prepared_test_manifest.json",
    test_manifest="train/data/manifests/prepared_test_manifest.json",
    num_workers=24,
    bs=config.bs,
)

model = QuartznetModule.load_from_nemo(
    checkpoint_name="QuartzNet15x5Base-En",
)

labels = [
    " ",
    "ɑ",
    "d",
    "m",
    "ɛ",
    "v",
    "ʒ",
    "ð",
    "ʃ",
    "θ",
    "g",
    "i",
    "ŋ",
    "b",
    "ɪ",
    "u",
    "ɹ",
    "z",
    "t",
    "ʌ",
    "j",
    "f",
    "p",
    "ɔ",
    "k",
    "s",
    "æ",
    "n",
    "h",
    "ʊ",
    "ʧ",
    "l",
    "w",
    "ʤ",
    "o",
    "X",
]

model.change_vocab(labels)
model.hparams.learning_rate = config.lr_frozen
model.hparams.betas = config.betas
callback_finetune = FinetuneEncoderDecoder(decoder_lr=config.lr_unfrozen, encoder_initial_lr_div=10)
wb_logger = pl_loggers.WandbLogger()
trainer = pl.Trainer(
    gpus=1,
    max_epochs=3,
    check_val_every_n_epoch=1,
    log_every_n_steps=1,
    num_sanity_val_steps=2,
    deterministic=True,
    logger=wb_logger,
    callbacks=[LogSpectrogramsCallback(), LogResultsCallback(), callback_finetune],
)
trainer.fit(model=model, datamodule=full_dm)
