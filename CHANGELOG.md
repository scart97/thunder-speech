# Changelog

<!--next-version-placeholder-->

## v3.2.0 (2022-08-04)
### Feature
* Bump minimum python from 3.7 to 3.8 ([`e11e05c`](https://github.com/scart97/thunder-speech/commit/e11e05cd61cbde357fb2fcf2d811583d76064729))
* **citrinet:** Add augmentation parameters ([`fa6ca23`](https://github.com/scart97/thunder-speech/commit/fa6ca233b2a306666c76a6a5ea133a457b758337))
* **quartznet:** Add augmentation parameters ([`292ad04`](https://github.com/scart97/thunder-speech/commit/292ad0482a8bc559cefe7f5ff5408fd20b714865))
* **citrinet:** Add dropout parameter ([`af755d1`](https://github.com/scart97/thunder-speech/commit/af755d1a57e86218e19980f3dfa44ca3d93fc778))
* **quartznet:** Add dropout parameter ([`35ecbdb`](https://github.com/scart97/thunder-speech/commit/35ecbdbff20f24e5b48fcf52d7833e73b93db690))
* Add SpecAugment and CutOut to Quartznet/Citrinet ([`aab6f89`](https://github.com/scart97/thunder-speech/commit/aab6f892b59b006e859a61f386d4628eb00cfc4f))
* **quartznet:** Add original SpecAugment and SpecCutout from nemo ([`c121de4`](https://github.com/scart97/thunder-speech/commit/c121de45b82c198f7054d637d9dad3f95cf0db53))

### Fix
* **huggingface:** Fix data2vec loading ([`fc91a9f`](https://github.com/scart97/thunder-speech/commit/fc91a9f8db37e5b0a442040fb15453a52b1409b1))
* **spec_augment:** Only augment while training ([`971c31c`](https://github.com/scart97/thunder-speech/commit/971c31c363ff2a5d95eb5b8c237df5c1249338a3))
* **spec_augment:** Masks should have a minimum length of 1 to be valid ([`4d45ac4`](https://github.com/scart97/thunder-speech/commit/4d45ac45f9c09be28d7e20ec516fd9ebb0ace764))

## v3.1.3 (2022-07-05)
### Fix
* **blocks:** Assert normalize_tensor input is properly masked ([`0643122`](https://github.com/scart97/thunder-speech/commit/0643122a32b08d6b801f13a9f0a720083d75d1af))

### Documentation
* **mkdocstrings:** Revert to python-legacy handler ([`405a9ae`](https://github.com/scart97/thunder-speech/commit/405a9ae792c57ebdadb272f00e4fbe7032612207))

## v3.1.2 (2022-05-11)
### Fix
* Transpose the output of huggingface encoders to be consistent with the convention used in the library ([`951307e`](https://github.com/scart97/thunder-speech/commit/951307e3a1479502700dedc22761eeb7b1fc44b2))
* **finetune:** Better error message in case of missing parameter ([`405325b`](https://github.com/scart97/thunder-speech/commit/405325b58396bee846db9138b8b96703188bb42c))

### Documentation
* Updating old docs ([`ae6a0b8`](https://github.com/scart97/thunder-speech/commit/ae6a0b8100370664aa909091767ad6f82803d169))

## v3.1.1 (2022-05-11)
### Fix
* **huggingface:** Fix compatibility with the huggingface tokenizer ([`42a8878`](https://github.com/scart97/thunder-speech/commit/42a8878a27ad0814e2e35afbd29c9fd597945c0e))

## v3.1.0 (2022-05-10)
### Feature
* **text_processing:** Add support to use custom tokenizers ([`36880c1`](https://github.com/scart97/thunder-speech/commit/36880c15343de2f25b25886e59e2e77e4e6a2855))

## v3.0.4 (2022-05-09)
### Fix
* Trigger release with updated dependencies ([`2da533f`](https://github.com/scart97/thunder-speech/commit/2da533fd9874dcfb7d106f4f82e18d9396f86c00))

## v3.0.3 (2022-04-18)
### Fix
* Improve typing on decoder and optimizer arguments ([`83c0758`](https://github.com/scart97/thunder-speech/commit/83c075897f7c38f92653e941860f40e8eb3d1e88))

## v3.0.2 (2022-04-18)
### Fix
* **huggingface:** Correct loading of checkpoints that have unused tokens ([`887608e`](https://github.com/scart97/thunder-speech/commit/887608e7f41ef47d71429ddb9211cbc4eb69d581))
* **huggingface:** Fix loading of pretrained only models, that lack the tokenizer ([`65106ee`](https://github.com/scart97/thunder-speech/commit/65106eea7dd6a4dc7c4f13b6a6d74567835b306e))

## v3.0.1 (2022-04-17)
### Fix
* Trigger release with updated dependencies ([`3b59ffe`](https://github.com/scart97/thunder-speech/commit/3b59ffe446183ccefb8229eebaca77a4e5e098df))

### Documentation
* **readme:** Update readme dev instructions ([`6e39bd9`](https://github.com/scart97/thunder-speech/commit/6e39bd9d99d61f7ec0a07fd39b732b7b17593c8e))
