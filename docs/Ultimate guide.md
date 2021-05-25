# The ultimate guide to speech recognition (WIP)

This guide has the purpose to give you all the steps necessary to achieve a decent (but not necessarily state-of-the-art) speech recognition system in a new language.


## Gathering the data

Speech recognition systems are really sensitive to the quality of data used to train them.
Also, they usually require from hundreads to thousands of hours depending on the quality expected.

Some good sources for data are [Mozilla commonvoice](https://commonvoice.mozilla.org/en/datasets), the [OpenSLR project](https://openslr.org/resources.php) or [Tatoeba](https://tatoeba.org/en/).

After you download some initial data, there's a number of data quality problems that are expected and need to be fixed if you want to increase the performance of the trained models.
First of all, list all of the audio files by increasing size and check if there's any corrupted file (usually they're very small).
Remove them from the training data.

Then install [sox](http://sox.sourceforge.net/), that's the best tool to inspect and convert audio files.
It should come with a basic tool to inspect any file in the terminal, called `soxi`. As an example:

```
$ soxi example_file.wav

Input File     : 'example_file.wav'
Channels       : 1
Sample Rate    : 16000
Precision      : 16-bit
Duration       : 00:00:04.27 = 94053 samples ~ 319.908 CDDA sectors
File Size      : 188k
Bit Rate       : 353k
Sample Encoding: 16-bit Signed Integer PCM
```

That's the usual format of files used in speech recognition research.
Wav files, encoded with a 16-bit PCM codec and a sample rate of 16 kHz.
The file format and codec can vary and will only affect the quality of the audio, but the sample rate is the essential one.
Trained models only work with a specific sample rate, and any file with a different one must be resampled either at the file
level or directly after loading with torchaudio.

Sox has more capabilities than just listing audio metadata.
It can read almost any file format and convert to others.
If you have an mp3 file at 44.1 kHz, and want to convert into the usual wav format above, you can use:

```
sox input_file.mp3 -r 16000 -c 1 -b 16 output_file.wav
```

The flags used represent:

* `-r 16000`: 16 kHz sample rate
* `-c 1`: convert to mono (1 channel)
* `-b 16`: convert to PCM 16-bit
* `output_file.wav`: Sox understand that the output will be wav just by the file extension

Ideally all of the training and inference audio files should have the same characteristics, so it's a good idea to transform them into a common format before training.
As the wav format does not have any compression, the resulting data will demand a huge HDD space.
If that's a problem, you can instead convert the files to mp3, that way you lose a small percentage of the performance but can achieve up to 10x smaller dataset sizes.

Now take a look at the labels. We are searching for a number of different problems here:

* Strange symbols: can easily find if you list all unique characters in the dataset
* Text in another language: remove these files
* Additional info that should not be there, like speaker identification as part of the transcription (common in subtitles)
* Regional/temporal differences that can cause the same words to have multiple written forms: mixing data from multiple countries that speak the same language, or using labels that came from old books

Try to fix those label problems, or remove them from the training set if you have lots of data.
Don't spend weeks just looking at the data, but have a small subset that you can trust is properly cleaned, even if that means manually labeling again.
After you train the first couple of models, it's possible to use the model itself to help find problems in the training data.

## Writing the dataset/datamodule

`TODO: fill this section with the nemo manifest example`

* load source
* load audio
* load text
* fix text
    * Expand contractions (`I'm` becomes `I am`)
    * Expand numbers (`42` becomes `forty two`)
    * Optionally remove punctuation
* datamodule with sources


## First train

For this first train, you should only try to overfit one batch.
This is the most simple test, and if you can't get past it then anything
more complex that you try will be wasted time.

To do it, try to load a training dataset with only one batch worth of data.
The validation/test sets can be as usual, you will ignore them at this step.
As we are using pytorch lightning, there's a trainer flag to limit the number of training batches (`limit_train_batches=1`) that can be used.
Also, remember to disable any shuffle at the dataloader, to ensure the same batch will be used every epoch.


Before you run the training, disable any augmentation, regularization and advanced stuff like learning rate scheduling. You can start with either a pretrained model or a clean new one,
but either way don't freeze any parameters, just let it all train.

Start the training, and you should see the loss follow a pattern where, the more time you let it run,
the final value will be lower. This means that small bumps will happen, but it will always recover
and keep going down. The ideal point is where you run the prediction on the batch that you overfit,
and the model doesn't make a single mistake.

Some problems that can happen:

* **The loss is negative**: There's a blank in the target text, find and remove it. Blanks should only be produced by the model, never at the labels.

* **There's no predictions at all**: let it train for more time

* **Still, there's no predictions after a long time**: Check if the target texts are being processed correctly. Inside the training step, decode the target text and assert that it returns what you expect

* **The loss does a 'U' curve where it starts normally but then turns around and just keep increasing**: try to lower the learning rate

## Second train

Now repeat the first training, but with around 10 hours of data.
This number depends on the hardware that you have available,
but something that gives you 2 minute epochs is a good amount.

This time, you're not trying to overfit anymore.
The validation loss will start to get lower,
and the metrics will improve compared to the first training.
But, quickly, the model will reach the point where the data is enough and it will start to overfit to the training data.

`TODO: better graphs?`

Expected train loss:

```
\
 \
  \
   \
    \______
```

Expected val loss/metrics:

```
\
 \
  \
   \      /
    \____/
```

## Scaling to the whole dataset

`TODO: expand this section`

* break long audios - more than 25s is usually bad
* Use the model to find problems
    * Sort by loss descending and manually check the files
    * Sort by CER descending and manually check the files
    * Sort by CER ascending on the validation/test set to find possible data leak
* Watch for the loss spikes during training

## Reducing overfit

`TODO: expand this section`

* fastai recipe
    * https://youtu.be/4u8FxNEDUeg?t=1333
        * Add more data
        * Add augmentation
        * Regularization

## Deploy!

`TODO: expand this section`

* torch jit
* Streamlit
* torchserve
* bentoml
