# Writing a custom data pipeline

There's a `BaseSpeechDataset` class that can be used as base to load the data.
The library expects that each element in the dataset will be a tuple (audio_tensor, text_label),
where the audio tensor has shape (channels, time) and text_label is the corresponding label as a string.

The `BaseSpeechDataset` has two important properties:
1. A list (or iterable) `.items`, that has all the metadata to load every item in the dataset
2. The `.loader` module. That is a pytorch class that uses torchaudio to load audio tensors and can apply resampling and mono conversion.
It was designed to be exported independently of the dataset, so that the same data loading can be used during inference.

To get each element in the dataset, the following code is used, and each function call can be overwritten to control functionality:

```python
class BaseSpeechDataset(Dataset):
    def __getitem__(self, index: int) -> Tuple[Tensor, str]:
        item = self.get_item(index)
        # Dealing with input
        audio, sr = self.open_audio(item)
        audio = self.preprocess_audio(audio, sr)
        # Dealing with output
        text = self.open_text(item)
        text = self.preprocess_text(text)
        return audio, text
```

The flow of loading the data happens as follows:

1. `self.get_item` is called with a specific index. It uses `self.items` to return the specific metadata to that example
2. All of the metadata is sent to `self.open_audio`. The relevand subset is used to load the audio tensor and corresponding sample rate, using `self.loader.open_audio(...)`
3. Inside `self.preprocess_audio` the audio tensor is resampled and converted to mono if necessary using `self.loader.preprocess_audio(...)`.
At this point, any augmentation that happens at the signal level to individual items can be applyed.
Only the audio tensor is returned, because it's assumed that every audio in the dataset will be resampled to the same sample rate
4. `self.open_text` uses the same metadata to open the corresponding text label
5. `self.preprocess_text` can be used to apply any transform directly to the text. Common options are lower case, expanding contractions (`I'm` becomes `I am`), expanding numbers (`42` becomes `forty two`) and removing punctuation

## Example: Loading data from nemo

This example will implement `thunder.data.datamodule.ManifestDatamodule` and `thunder.data.dataset.ManifestSpeechDataset`.

### Load source

The nemo manifest file contains one json in each line, with the relevant data:

```
{"audio_filepath": "commonvoice/pt/train/22026127.mp3", "duration": 4.32, "text": "Quatro"}
{"audio_filepath": "commonvoice/pt/train/23920071.mp3", "duration": 2.256, "text": "Oito"}
{"audio_filepath": "commonvoice/pt/train/20272843.mp3", "duration": 2.544, "text": "Eu vou desligar"}
```

We can load this using the stdlib `json` and `pathlib` modules:

```python
from pathlib import Path
import json

file = Path("manifest.json")
# Reading from the manifest file
items = [json.loads(line) for line in file.read_text().strip().splitlines()]
```

The result is a list, where each element is a dictionary with the relevant data to a single example in the dataset.
Let's start to wrap this code inside a `BaseSpeechDataset`:

```python
from pathlib import Path
import json
from thunder.data.dataset import BaseSpeechDataset

class ManifestSpeechDataset(BaseSpeechDataset):
    def __init__(self, file: Union[str, Path], force_mono: bool, sample_rate: int):
        file = Path(file)
        items = [json.loads(line) for line in file.read_text().strip().splitlines()]
        super().__init__(items, force_mono=force_mono, sample_rate=sample_rate)
```

### Load audio

We know that the "audio_filepath" key is related to the input:

```python
from pathlib import Path
import json
from thunder.data.dataset import BaseSpeechDataset

class ManifestSpeechDataset(BaseSpeechDataset):
    def __init__(self, file: Union[str, Path], force_mono: bool, sample_rate: int):
        file = Path(file)
        items = [json.loads(line) for line in file.read_text().strip().splitlines()]
        super().__init__(items, force_mono=force_mono, sample_rate=sample_rate)

    def open_audio(self, item: dict) -> Tuple[Tensor, int]:
        return self.loader.open_audio(item["audio_filepath"])
```

### Load text

The text is already loaded inside the "text" key:

```python
from pathlib import Path
import json
from thunder.data.dataset import BaseSpeechDataset

class ManifestSpeechDataset(BaseSpeechDataset):
    def __init__(self, file: Union[str, Path], force_mono: bool, sample_rate: int):
        file = Path(file)
        items = [json.loads(line) for line in file.read_text().strip().splitlines()]
        super().__init__(items, force_mono=force_mono, sample_rate=sample_rate)

    def open_audio(self, item: dict) -> Tuple[Tensor, int]:
        return self.loader.open_audio(item["audio_filepath"])

    def open_text(self, item: dict) -> str:
        return item["text"]
```

### Fix text

The only text processing that will be applied in this example is transforming all the characters to lowercase:

```python
from pathlib import Path
import json
from thunder.data.dataset import BaseSpeechDataset

class ManifestSpeechDataset(BaseSpeechDataset):
    def __init__(self, file: Union[str, Path], force_mono: bool, sample_rate: int):
        file = Path(file)
        items = [json.loads(line) for line in file.read_text().strip().splitlines()]
        super().__init__(items, force_mono=force_mono, sample_rate=sample_rate)

    def open_audio(self, item: dict) -> Tuple[Tensor, int]:
        return self.loader.open_audio(item["audio_filepath"])

    def open_text(self, item: dict) -> str:
        return item["text"]

    def preprocess_text(self, text: str) -> str:
        return text.lower()
```


### Datamodule with sources

Just wrap the datasets inside a `BaseDataModule`.
Implement `get_dataset` to return the dataset for each split.

```python
from thunder.data.datamodule import BaseDataModule

class ManifestDatamodule(BaseDataModule):
    def __init__(
        self,
        train_manifest: str,
        val_manifest: str,
        test_manifest: str,
        force_mono: bool = True,
        sample_rate: int = 16000,
        batch_size: int = 10,
        num_workers: int = 8,
    ):
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self.manifest_mapping = {
            "train": train_manifest,
            "valid": val_manifest,
            "test": test_manifest,
        }
        self.force_mono = force_mono
        self.sample_rate = sample_rate

    def get_dataset(self, split: str) -> ManifestSpeechDataset:
        return ManifestSpeechDataset(
            self.manifest_mapping[split], self.force_mono, self.sample_rate
        )
```

### Using the datamodule

```python
datamodule = ManifestDatamodule("train_manifest.json", "val_manifest.json", "test_manifest.json", batch_size = 32)
```
