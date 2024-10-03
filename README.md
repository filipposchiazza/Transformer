# GPT Model for image latent codes generation
Transfomers are powerful models that have been used for a variety of tasks, especially in the field of natural language processing. However, they can also be used for other tasks, such as generating image latent codes. This project is based on a minimal GPT model implementation (https://github.com/karpathy/minGPT) used for creating a Latent Transformer model that can generate image latent codes. These codes can be obtained from a pretrained encoder model (in this specific case a Vector Quantized Variational Autoencoder model) and then used to generate images. 

## Repository Structure
The files are organized as follows:
- `mingpt.py`: Contains a minimal implementation of the GPT model.
- `latent_transformer.py`: Contains the implementation of the Latent Transformer model, used to encoder images in codes and apply the GPT model to generate new codes.
- `transformer_trainer.py`: Contains the training object implementation for the Latent Transformer model.
- `train.py`: Contains the training script for the Latent Transformer model.
- `generation.py`: Contains the generation script for the Latent Transformer model and the decoding process to obtain images from the generated codes.
- `config.py`: Contains the configuration for the training and generation scripts.

## How to use for training
Import the necessary dependencies:
```python
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
```

Load the image dataset:
```python
# Load image data
train_dataset, val_dataset, train_dataloader, val_dataloader = prepare_ImageDataset(img_dir=config.IMG_DIR, 
                                                                                    batch_size=config.BATCH_SIZE,
                                                                                    validation_split=config.VALIDATION_SPLIT,
                                                                                    transform=config.TRANSFORM, 
                                                                                    fraction=config.FRACTION,
										                                            subpatch=config.SUBPATCH)

```

Load the pretrained VQVAE model:
```python
pretrained_model = VQVAE.load_model(config.PRETRAINED_MODEL_PATH).to(config.DEVICE)
```

Create GPT model
```python
gpt_model = GPT(vocab_size=config.VOCAB_SIZE,
                emb_dim=config.EMB_DIM,
                num_heads=config.NUM_HEADS,
                num_layers=config.NUM_LAYERS,
                block_size=config.BLOCK_SIZE,
                emb_dropout=config.EMB_DROPOUT,
                attention_dropout=config.ATTENTION_DROPOUT,
                residual_dropout=config.RESIDUAL_DROPOUT).to(config.DEVICE)
```

Create the latent transformer, the optimizer and the trainer:
```python
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
```

Train the model
```python
history = trainer.train(train_dataloader=train_dataloader,
                        num_epochs=config.NUM_EPOCHS,
                        device=config.DEVICE,
                        validation_dataloader=val_dataloader)
```

Save the model and the history
```python
# Save model
latent_transformer.save_model(config.SAVE_FOLDER)

# Save history
trainer.save_history(config.SAVE_FOLDER)
```



## How to use for generation
Import the necessary dependencies:
```python
import sys
import config
sys.path.append(config.PRETRAINED_MODEL_PARAMETERS_PATH)
sys.path.append(config.PRETRAINED_MODEL_PATH)
import torch
import matplotlib.pyplot as plt
from latent_transformer import LatentTransformer
from vqvae import VQVAE
```

Set the number of images to generate and the SOS token:
```python
num_imgs = 10
sos_token = torch.zeros(num_imgs, 1)
```

Load the pretrained model and the latent transformer:
```python
pretrained_model = VQVAE.load_model(config.PRETRAINED_MODEL_PATH).to(config.DEVICE)

latent_transformer = LatentTransformer.load_model(config.SAVE_FOLDER, pretrained_model)
```

Generate the latent codes and decode them to obtain the images:
```python
gen_codes = latent_transformer.sample(sos_token, max_len=256).unsqueeze(1)
idx_to_vec = pretrained_model.vq_layer.emb(gen_codes).squeeze(1)
idx_to_vec = idx_to_vec.permute(0, 3, 1, 2)
imgs = pretrained_model.decoder(idx_to_vec)
```

Plot the generated images:
```python
for i in range(num_imgs):
    plt.imshow(imgs[i].permute(1, 2, 0).detach())
    plt.show()
```


## Dependencies
* python == 3.12
* pytorch == 2.4.1
* torchvision == 0.19.1 
* tqdm == 4.66.5
* numpy == 1.26.4
* matplotlib == 3.9.2