import sys
import config
sys.path.append(config.PRETRAINED_MODEL_PARAMETERS_PATH)
sys.path.append(config.PRETRAINED_MODEL_PATH)
import torch
from latent_transformer import LatentTransformer
from transformer_trainer import LatentTransformerTrainer
from mingpt import GPT
from vqvae import VQVAE
from dataset import prepare_ImageDataset


# Load image data
train_dataset, val_dataset, train_dataloader, val_dataloader = prepare_ImageDataset(img_dir=config.IMG_DIR, 
                                                                                    batch_size=config.BATCH_SIZE,
                                                                                    validation_split=config.VALIDATION_SPLIT,
                                                                                    transform=config.TRANSFORM, 
                                                                                    fraction=config.FRACTION,
										                                            subpatch=config.SUBPATCH)

# Load pretrained VQVAE model       
pretrained_model = VQVAE.load_model(config.PRETRAINED_MODEL_PATH).to(config.DEVICE)

# Create GPT model
gpt_model = GPT(vocab_size=config.VOCAB_SIZE,
                emb_dim=config.EMB_DIM,
                num_heads=config.NUM_HEADS,
                num_layers=config.NUM_LAYERS,
                block_size=config.BLOCK_SIZE,
                emb_dropout=config.EMB_DROPOUT,
                attention_dropout=config.ATTENTION_DROPOUT,
                residual_dropout=config.RESIDUAL_DROPOUT).to(config.DEVICE)

# Create latent transformer
latent_transformer = LatentTransformer(pretrained_model=pretrained_model,
                                       gpt_model=gpt_model,
                                       sos_token=config.SOS_TOKEN,
                                       pkeep=config.PKEEK,
                                       device=config.DEVICE)

# Create optimizer
optimizer = torch.optim.Adam(latent_transformer.parameters(), lr=config.LEARNING_RATE)

# Create trainer
trainer = LatentTransformerTrainer(latent_transformer=latent_transformer,
                                   optimizer=optimizer)

# Train the model
history = trainer.train(train_dataloader=train_dataloader,
                        num_epochs=config.NUM_EPOCHS,
                        device=config.DEVICE,
                        validation_dataloader=val_dataloader)

# Save model
latent_transformer.save_model(config.SAVE_FOLDER)

# Save history
trainer.save_history(config.SAVE_FOLDER)


