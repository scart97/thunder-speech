[![codecov](https://codecov.io/gh/scart97/thunder-speech/branch/master/graph/badge.svg?token=USCEGEGM3D)](https://codecov.io/gh/scart97/thunder-speech)
![Test](https://github.com/scart97/thunder-speech/workflows/Test/badge.svg)
[![docs](https://img.shields.io/badge/docs-read-informational)](https://scart97.github.io/thunder-speech/)

# Thunder speech

> A Hackable speech recognition library.

What to expect from this project:

- End-to-end speech recognition models
- Simple fine-tuning to new languages
- Inference support as a first-class feature
- Developer oriented api

What it's not:

- A general-purpose speech toolkit
- A collection of complex systems that require thousands of gpu-hours and expert knowledge, only focusing on the state-of-the-art results


## Quick usage guide

### Install

Install the library from PyPI:

```
pip install thunder-speech
```


### Load the model and train it

```py
from thunder.registry import load_pretrained
from thunder.quartznet.compatibility import QuartznetCheckpoint

# Tab completion works to discover other QuartznetCheckpoint.*
module = load_pretrained(QuartznetCheckpoint.QuartzNet5x5LS_En)
# It also accepts the string identifier
module = load_pretrained("QuartzNet5x5LS_En")
# Or models from the huggingface hub
module = load_pretrained("facebook/wav2vec2-large-960h")
```

### Export to a pure pytorch model using torchscript

```py
module.to_torchscript("model_ready_for_inference.pt")

# Optional step: also export audio loading pipeline
from thunder.data.dataset import AudioFileLoader

loader = AudioFileLoader(sample_rate=16000)
scripted_loader = torch.jit.script(loader)
scripted_loader.save("audio_loader.pt")
```

### Run inference in production

``` python
import torch
import torchaudio

model = torch.jit.load("model_ready_for_inference.pt")
loader = torch.jit.load("audio_loader.pt")
# Open audio
audio = loader("audio_file.wav")
# transcriptions is a list of strings with the captions.
transcriptions = model.predict(audio)
```

## More quick tips

If you want to know how to access the raw probabilities and decode manually or fine-tune the models you can access the documentation [here](https://scart97.github.io/thunder-speech/quick%20reference%20guide/).

## Contributing

The first step to contribute is to do an editable installation of the library:

```
git clone https://github.com/scart97/thunder-speech.git
cd thunder-speech
poetry install
pre-commit install
```

Then, make sure that everything is working. You can run the test suit, that is based on pytest:

```
RUN_SLOW=1 poetry run pytest
```

Here the `RUN_SLOW` flag is used to run all the tests, including the ones that might download checkpoints or do small training runs and are marked as slow. If you don't have a CUDA capable gpu, some tests will be unconditionally skipped.


## Influences

This library has heavy influence of the best practices in the pytorch ecosystem.
The original model code, including checkpoints, is based on the NeMo ASR toolkit.
From there also came the inspiration for the fine-tuning and prediction api's.

The data loading and processing is loosely based on my experience using fast.ai.
It tries to decouple transforms that happen at the item level from the ones that are efficiently implemented for the whole batch at the GPU.
Also, the idea that default parameters should be great.

The overall organization of code and decoupling follows the pytorch-lightning ideals, with self-contained modules that try to reduce the boilerplate necessary.

Finally, the transformers library inspired the simple model implementations, with a clear separation in folders containing the specific code that you need to understand each architecture and preprocessing, and their strong test suit.
