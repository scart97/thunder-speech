# Quick reference guide

## How to export a Quartznet .nemo file to a pure pytorch model?

```py
from thunder.quartznet.module import QuartznetModule

module = QuartznetModule.load_from_nemo(
    nemo_filepath="/path/to/checkpoint.nemo"
)
module.to_torchscript("model_ready_for_inference.pt")
```


## How to run inference on that exported file?


``` python
import torch
import torchaudio

model = torch.jit.load("model_ready_for_inference.pt")
audio, sr = torchaudio.load("audio_file.wav")
# Optionally resample if sr is different from original model sample rate
# tfm = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
# audio = tfm(audio)
transcriptions = model.predict(audio)
# transcriptions is a list of strings with the captions.
```

??? note
    The exported model only depends on pytorch and torchaudio, and the later is only used
    to open the audio file into a tensor. If torchaudio.load could be compiled inside the
    model in the future, similar to what already happens with torchvision, the dependency
    can be removed and only the base pytorch will be necessary to run inferece!


## What if I want the probabilities instead of the captions?

Instead of `model.predict(audio)`, use just `model(audio)`

``` python hl_lines="6"
import torch
import torchaudio

model = torch.jit.load("model_ready_for_inference.pt")
audio, sr = torchaudio.load(audio_name)
probs = model(audio)
# If you also want the transcriptions:
transcriptions = model.text_transform.decode_prediction(probs.argmax(1))
```


## How to finetune a model if I already have the nemo manifests prepared?

``` python
import pytorch_lightning as pl

from thunder.data.datamodule import ManifestDatamodule
from thunder.quartznet.module import QuartznetModule,  NemoCheckpoint

dm = ManifestDatamodule(
    train_manifest="/path/to/train_manifest.json",
    val_manifest="/path/to/val_manifest.json",
    test_manifest="/path/to/test_manifest.json",
)
# Tab completion works to discover other Nemocheckpoint.*
model = QuartznetModule.load_from_nemo(checkpoint_name=NemoCheckpoint.QuartzNet5x5LS_En)

trainer = pl.Trainer(
    gpus=-1, # Use all gpus
    max_epochs=10,
)

trainer.fit(model=model, datamodule=dm)
```

!!! danger
    This will probably have a subpar result right now, as I'm still working on
    properly fine tuning (freeze encoder at the start, learning rate scheduling,
    better defaults)


## How to get the initial_vocab_tokens from my dataset?

```python
from thunder.text_processing.tokenizer import char_tokenizer, get_most_frequent_tokens

my_datamodule = CustomDatamodule(...)
my_datamodule.prepare_data()
my_datamodule.setup(None)

train_corpus = " ".join(my_datamodule.train_dataset.all_outputs())
initial_vocab_tokens = get_most_frequent_tokens(train_corpus, char_tokenizer)
```
