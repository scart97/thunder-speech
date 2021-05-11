import torch
from pytorch_lightning.callbacks import Callback
import wandb


class LogSpectrogramsCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        batch = next(iter(trainer.train_dataloader))
        features = self._generate_spectrograms(pl_module, batch)
        images = [wandb.Image(features[i].unsqueeze(0)) for i in range(features.shape[0])]
        wandb.log({"spectrograms": images})

    def _generate_spectrograms(self, pl_module, batch):
        audio, audio_lens, texts = batch
        y, y_lens = pl_module.text_pipeline.encode(texts, device=pl_module.device)
        max_samples = audio.shape[-1]
        seq_lens = torch.floor((audio_lens * max_samples) / pl_module.n_window_stride) + 1
        features = pl_module.features(audio.cuda(), seq_lens.cuda())
        return features


class LogResultsCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        batch = next(iter(trainer.val_dataloaders[0]))
        audio, audio_lens, texts = batch
        y, y_lens = pl_module.text_pipeline.encode(texts, device=pl_module.device)
        probabilities, encoded_lens = pl_module(audio.cuda(), audio_lens.cuda())
        decoded_preds = pl_module.text_pipeline.decode_prediction(probabilities.argmax(1))
        decoded_targets = pl_module.text_pipeline.decode_prediction(y)
        self._log_results(audio, decoded_targets, decoded_preds)

    def _log_results(self, audio, decoded_targets, decoded_preds, num_examples=4):
        audios = []
        examples = list(zip(audio, decoded_targets, decoded_preds))[:num_examples]
        for (waveform, targ, pred) in examples:
            caption = f"Targ:{targ} --- Pred:{pred}\n"
            audios.append(wandb.Audio(waveform.cpu(), caption=caption, sample_rate=16000))
        wandb.log({"examples": audios})
