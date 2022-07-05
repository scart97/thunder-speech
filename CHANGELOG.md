# Changelog

<!--next-version-placeholder-->

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
