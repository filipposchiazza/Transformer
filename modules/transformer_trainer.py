import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import pickle


class LatentTransformerTrainer(nn.Module):

    def __init__(self, 
                 latent_transformer,
                 optimizer):
        """ Trainer for the latent transformer model.

        Parameters
        ----------
        latent_transformer : LatentTransformer
            Latent transformer model.
        optimizer : torch.optim
            Optimizer.
        """
        super(LatentTransformerTrainer, self).__init__()
        self.model = latent_transformer
        self.optimizer = optimizer



    def train(self, 
              train_dataloader,
              num_epochs,
              device,
              validation_dataloader=None):
        """ Train the latent transformer model.

        Parameters
        ----------
        train_dataloader : torch.utils.data.DataLoader
            Training data loader.
        num_epochs : int
            Number of epochs.
        device : str
            Device to use.
        validation_dataloader : torch.utils.data.DataLoader
            Validation data loader.

        Returns
        -------
        history : dict
            Training history.
        """

        self.history = {'train_loss': [], 
                        'val_loss': []}

        for epoch in range(num_epochs):

            # Train one epoch
            self.model.train()
            train_loss = self._train_one_epoch(train_dataloader, epoch, device)
            self.history['train_loss'].append(train_loss)

            # Validate
            if validation_dataloader is not None:
                self.model.eval()
                val_loss = self._validate(validation_dataloader, device)
                self.history['val_loss'].append(val_loss)

        return self.history



    def _train_one_epoch(self,
                         train_dataloader,
                         epoch,
                         device):
        """ Train the model for one epoch.

        Parameters
        ----------
        train_dataloader : torch.utils.data.DataLoader
            Training data loader.
        epoch : int
            Current epoch.
        device : str
            Device to use.

        Returns
        -------
        mean_loss : float
            Mean loss value.
        """
        running_loss = 0.0
        mean_loss = 0.0

        with tqdm(train_dataloader, unit='batches') as tepoch:
            for batch_idx, imgs in enumerate(tepoch):

                # Update the progress bar description
                tepoch.set_description(f'Epoch {epoch+1}')

                # Load data to device
                imgs = imgs.to(device)

                # forward step
                self.optimizer.zero_grad()
                logits, targets = self.model(imgs)
                
                # Compute loss
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))   # reshape done for crossentropy

                # backward step
                loss.backward()
                self.optimizer.step()

                # Update loss values
                running_loss += loss.item()
                mean_loss = running_loss / (batch_idx + 1)

                # Update the progress bar
                tepoch.set_postfix({'Loss': '{:.3f}'.format(mean_loss)})

        return mean_loss



    @torch.no_grad()           
    def _validate(self,
                  validation_dataloader,
                  device):
        """ Validate the model.

        Parameters
        ----------
        validation_dataloader : torch.utils.data.DataLoader
            Validation data loader.
        device : str
            Device to use.

        Returns
        -------
        mean_val_loss : float
            Mean validation loss value.
        """
        running_val_loss = 0.0

        for batch_idx, imgs in enumerate(validation_dataloader):
                
                # Load data to device
                imgs = imgs.to(device)
    
                # forward step
                logits, targets = self.model(imgs)
    
                # Compute loss
                val_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

                # Update loss values
                running_val_loss += val_loss.item()
        
        mean_val_loss = running_val_loss / len(validation_dataloader)
        print(f'Validation loss: {mean_val_loss:.3f}')

        return mean_val_loss



    def save_history(self, save_folder):
        filename = os.path.join(save_folder, 'latent_transformer_history.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(self.history, f)
        
