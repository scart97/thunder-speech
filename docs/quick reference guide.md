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
transcriptions = model.text_pipeline.decode_prediction(probs)
```
