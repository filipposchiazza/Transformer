import sys
import config
sys.path.append(config.PRETRAINED_MODEL_PARAMETERS_PATH)
sys.path.append(config.PRETRAINED_MODEL_PATH)
import torch
import matplotlib.pyplot as plt
from latent_transformer import LatentTransformer
from vqvae import VQVAE



num_imgs = 10
sos_token = torch.zeros(num_imgs, 1)

pretrained_model = VQVAE.load_model(config.PRETRAINED_MODEL_PATH).to(config.DEVICE)

latent_transformer = LatentTransformer.load_model(config.SAVE_FOLDER, pretrained_model)

gen_codes = latent_transformer.sample(sos_token, max_len=256).unsqueeze(1)

idx_to_vec = pretrained_model.vq_layer.emb(gen_codes).squeeze(1)

idx_to_vec = idx_to_vec.permute(0, 3, 1, 2)

imgs = pretrained_model.decoder(idx_to_vec)

for i in range(num_imgs):
    plt.imshow(imgs[i].permute(1, 2, 0).detach())
    plt.show()
