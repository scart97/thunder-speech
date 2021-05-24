[![codecov](https://codecov.io/gh/scart97/thunder-speech/branch/master/graph/badge.svg?token=USCEGEGM3D)](https://codecov.io/gh/scart97/thunder-speech)
![Test](https://github.com/scart97/thunder-speech/workflows/Test/badge.svg)
[![docs](https://img.shields.io/badge/docs-read-informational)](https://scart97.github.io/thunder-speech/)

# Thunder speech

> A Hackable speech recognition library.

What to expect from this project:

- End-to-end speech recognition models
- Simple fine tuning to new languages
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

### Import desired models

```py
from thunder.quartznet.module import QuartznetModule,  NemoCheckpoint

# Tab completion works to discover other Nemocheckpoint.*
model = QuartznetModule.load_from_nemo(checkpoint_name = NemoCheckpoint.QuartzNet5x5LS_En)
```
### Load audio and predict

```py
import torchaudio
audio, sr = torchaudio.load("my_sample_file.wav")

transcriptions = model.predict(audio)
# transcriptions is a list of strings with the captions.
```


## More quick tips

If you want to know how to export the models using torchscript, access the raw probabilities and decode manually or fine-tune the models you can access the documentation [here](https://scart97.github.io/thunder-speech/quick%20reference%20guide/).

## Contributing

The first step to contribute is to do a editable install of the library:

```
git clone https://github.com/scart97/thunder-speech.git
cd thunder-speech
pip install -e .[dev,testing]
pre-commit install
```

Then, make sure that everything is working. You can run the test suit, that is based on pytest:

```
RUN_SLOW=1 pytest
```

Here the `RUN_SLOW` flag is used to run all of the tests, including the ones that might download checkpoints or do small training runs and are marked as slow. If you don't have a CUDA capable gpu, some of the tests will be unconditionally skipped.


## Influences

This library has heavy influence of the best practices in the pytorch ecosystem.
The original model code, including checkpoints, is based on the NeMo ASR toolkit.
From there also came the inspiration for the fine-tuning and prediction api's.

The data loading and processing is loosely based on my experience using fast.ai.
It tries to decouple transforms that happen at the item level from the ones that are efficiently implemented for the whole batch at the GPU.
Also the idea that default parameters should be great.

The overall organization of code and decoupling follows the pytorch-lightning ideals, with self contained modules that try to reduce the boilerplate necessary.

Finally the transformers library inspired the simple model implementations, with a clear separation in folders containing the specific code that you need to understand each architecture and preprocessing. Also their strong test suit.


## Note

This project has been set up using PyScaffold 3.3. For details and usage
information on PyScaffold see https://pyscaffold.org/.
