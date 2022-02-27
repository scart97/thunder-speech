# Quick reference guide

## How to load a Quartznet/Citrinet .nemo file?

```py
from thunder.quartznet.compatibility import load_quartznet_checkpoint
from thunder.citrinet.compatibility import load_citrinet_checkpoint

module = load_quartznet_checkpoint("/path/to/quartznet.nemo")
module = load_citrinet_checkpoint("/path/to/citrinet.nemo")
```

## How to export models with special restrictions?

Case 1: Using Quartznet or Citrinet on platforms that doesnt support FFT (android, onnx):

```py
from thunder.registry import load_pretrained
from thunder.quartznet.transform import patch_stft
import torch

module = load_pretrained("QuartzNet5x5LS_En")
module.audio_transform = patch_stft(module.audio_transform)
module.to_torchscript("model_ready_for_inference.pt")
```

Case 2: Wav2vec 2.0 using torchscript


```py
from thunder.registry import load_pretrained
from thunder.huggingface.compatibility import prepare_scriptable_wav2vec

module = load_pretrained("facebook/wav2vec2-large-960h")
module = prepare_scriptable_wav2vec(module)
module.to_torchscript("model_ready_for_inference.pt")
```

## What if I want the probabilities instead of the captions during inference?

Instead of `model.predict(audio)`, use just `model(audio)`

``` python hl_lines="8"
import torch
import torchaudio

model = torch.jit.load("model_ready_for_inference.pt")
loader = torch.jit.load("audio_loader.pt")
# Open audio
audio = loader("audio_file.wav")
probs = model(audio)
# If you also want the transcriptions:
transcriptions = model.text_transform.decode_prediction(probs.argmax(1))
```


## How to finetune a model if I already have the nemo manifests prepared?

``` python
import pytorch_lightning as pl

from thunder.data.datamodule import ManifestDatamodule
from thunder.registry import load_pretrained
from thunder.callbacks import FinetuneEncoderDecoder

dm = ManifestDatamodule(
    train_manifest="/path/to/train_manifest.json",
    val_manifest="/path/to/val_manifest.json",
    test_manifest="/path/to/test_manifest.json",
)

model = load_pretrained("QuartzNet5x5LS_En")

trainer = pl.Trainer(
    gpus=-1, # Use all gpus
    max_epochs=10,
    callbacks=[FinetuneEncoderDecoder(unfreeze_encoder_at_epoch=1)],
)

trainer.fit(model=model, datamodule=dm)
```

## How to get the tokens from my dataset?

```python
from thunder.text_processing.tokenizer import char_tokenizer, get_most_frequent_tokens

my_datamodule = CustomDatamodule(...)
my_datamodule.prepare_data()
my_datamodule.setup(None)

train_corpus = " ".join(my_datamodule.train_dataset.all_outputs())
tokens = get_most_frequent_tokens(train_corpus, char_tokenizer)
```
