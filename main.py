import torch
import logging
import warnings
import torchmetrics
from torch import nn
from config import get_config
import lightning.pytorch as pl
from train import get_ds, get_model
from torch.nn import functional as F
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torchmetrics.text import CharErrorRate, WordErrorRate, BLEUScore


logger = logging.getLogger("Transformer")
logger.setLevel(level=logging.INFO)
fileHandler = logging.FileHandler(filename='prediction.log')
fileHandler.setLevel(level=logging.INFO)
logger.addHandler(fileHandler)

class TransformerLightning(pl.LightningModule):
  def __init__(self, config, tokenizer_src, tokenizer_tgt, label_smoothing=0.1):
    super().__init__()
    self.expected = []
    self.predicted = []
    self.initial_epoch = 0
    self.config = config
    self.tokenizer_src = tokenizer_src
    self.tokenizer_tgt = tokenizer_tgt
    self.model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size())
    self.loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=label_smoothing)
    self.train_loss = []
    self.save_hyperparameters()

  def forward(self, x):
    return self.model(x)
  
  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), 
                                 lr=self.config['lr'], 
                                 eps=1e-9)
    return optimizer


  def training_step(self, batch, batch_idx):
    encoder_input = batch['encoder_input'] # (b, seq_len)
    decoder_input = batch['decoder_input'] # (b, seq_len)
    encoder_mask = batch['encoder_mask'] # (b, 1, 1, seq_len)
    decoder_mask = batch['decoder_mask'] # (b, 1, seq_len, seq_len)

    # Run the tensors through the encoder, decoder and the projection layer
    encoder_output = self.model.encode(encoder_input, encoder_mask) # (b, seq_len, d_model)
    decoder_output = self.model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
    proj_output = self.model.project(decoder_output) # ( b, seq_len, vocab_size)

    # Compare the output with the label
    label = batch['label'] # (b, seq_len)

    # Compute the loss using cross entropy
    loss = self.loss_fn(proj_output.view(-1, self.tokenizer_tgt.get_vocab_size()), label.view(-1))
    self.train_loss.append(loss)

    self.log('loss', loss.item(), prog_bar=True, on_step=True, logger=True)


    return loss

  def validation_step(self, batch, batch_idx):
    encoder_input = batch['encoder_input'] # (b, seq_len)
    encoder_mask = batch['encoder_mask'] # (b, 1, 1, seq_len)

    assert encoder_input.size(
                0
            ) == 1, "Batch Size must be 1 for validation"
    
    model_out = self.greedy_decode(encoder_input, encoder_mask)

    source_text = batch["src_text"][0]
    target_text = batch["tgt_text"][0]
    model_out_text = self.tokenizer_tgt.decode(model_out.detach().cpu().numpy())

    logger.info(f"SOURCE - {source_text}")
    logger.info(f"TARGET - {target_text}")
    logger.info(f"PREDICTED - {model_out_text}")
    logger.info("=============================================================")

    self.expected.append(target_text)
    self.predicted.append(model_out_text)

  def on_validation_epoch_end(self):
    metric = CharErrorRate()
    cer = metric(self.predicted, self.expected)
    self.log('validation_cer', cer, prog_bar=True, on_epoch=True, logger=True)


    # Compute the word error rate
    metric = WordErrorRate()
    wer = metric(self.predicted, self.expected)
    self.log('validation_wer', wer, prog_bar=True, on_epoch=True, logger=True)

    # Compute the BLEU metric
    metric = BLEUScore()
    bleu = metric(self.predicted, self.expected)
    self.log('validation_bleu', bleu, prog_bar=True, on_epoch=True, logger=True)

    self.expected.clear()
    self.predicted.clear()

  def on_train_epoch_end(self):
    self.log('loss', torch.stack(self.train_loss).mean(), on_epoch=True, logger=True)
    print(f"Loss Mean - {torch.stack(self.train_loss).mean()}")
    self.train_loss.clear()

  def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
    optimizer.zero_grad(set_to_none=True)

  def greedy_decode(self, source, source_mask):
    sos_idx = self.tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = self.tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = self.model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source)

    while True:
        if decoder_input.size(1) == self.config['seq_len']:
            break

        # Build mask for target
        decoder_mask = self.causal_mask(decoder_input.size(1)).type_as(source_mask)

        # Calculate the output
        out = self.model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # Get next token
        prob = self.model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item())], 
            dim=1
        )

        if next_word == eos_idx:
            break

    return(decoder_input.squeeze(0))


  def causal_mask(self, size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return(mask == 0)
  

# training
trainer = pl.Trainer(
                     log_every_n_steps=1,
                     limit_val_batches=2,
                     check_val_every_n_epoch=1,
                     enable_model_summary=True,
                     max_epochs=20, 
                     accelerator='auto',
                     devices='auto',
                     strategy='auto',
                     logger=[TensorBoardLogger("logs/", name="transformer-scratch")],
                     callbacks=[ModelCheckpoint(
                                mode='min',
                                verbose=True,
                                monitor='validation_wer',
                                filename='transformer-{epoch:02d}-{validation_wer:.6f}', 
                                save_top_k=3)],
                     )
  

def main(config):
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = TransformerLightning(config, tokenizer_src, tokenizer_tgt)
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    cfg = get_config()
    cfg['batch_size'] = 16
    cfg['preload'] = None
    cfg['num_epochs'] = 20
    main(cfg)