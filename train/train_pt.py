import pytorch_lightning as pl
from thunder.data.datamodule import ManifestDatamodule
from thunder.quartznet.module import QuartznetModule
import wandb

pl.utilities.seed.seed_everything(8)
learning_rate = 3e-4

hyperparameter_defaults = dict(
    use_sampler=False,
    shuffle=True,
    original_fb_norm=False,
    original_qn_block=False,
    apply_final_pad=True,
    htk=False,
    use_encoded_lens=True,
)


wandb.init(config=hyperparameter_defaults, project="thunder-speech", entity="madeupmasters")
config = wandb.config
config.learning_rate = learning_rate
pt_dm = ManifestDatamodule(
    train_manifest="train/data/manifests/cv_train_pt.json",
    # train_manifest="train/data/manifests/debug_train_manifest.json",
    val_manifest="train/data/manifests/debug_test_manifest.json",
    test_manifest="train/data/manifests/debug_test_manifest.json",
    num_workers=48,
)

model = QuartznetModule.load_from_nemo(
    checkpoint_name="QuartzNet15x5Base-En",
)

model.train()

labels = [
    " ",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
]

model.change_vocab(labels)
# Freeze model
for param in model.encoder.parameters():
    param.requires_grad = False

model.hparams.learning_rate = learning_rate

trainer = pl.Trainer(
    gpus=1,
    max_epochs=1,
    check_val_every_n_epoch=2,
    log_every_n_steps=1,
    num_sanity_val_steps=0,
    deterministic=True,
)

trainer.fit(model=model, datamodule=pt_dm)
