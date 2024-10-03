import torch
import torch.nn as nn
import torch.nn.functional as F
from mingpt import GPT
import os
import pickle
from tqdm import tqdm



class LatentTransformer(nn.Module):

    def __init__(self,
                 pretrained_model, 
                 gpt_model,
                 sos_token,
                 pkeep,
                 device='cpu'):
        """ Latent transformer model. Use a pretrained VQVAE model to encode images and a GPT model to generate new codes.

        Parameters
        ----------
        pretrained_model : VQVAE
            Pretrained VQVAE model.
        gpt_model : GPT
            GPT model.
        sos_token : int
            Start of sentence token.
        pkeep : float
            Probability of keeping a token.
        device : str
            Device to use.
        """
        super(LatentTransformer, self).__init__()
        self.pretrained_model = pretrained_model.to(device)
        self.gpt_model = gpt_model.to(device)
        self.sos_token = sos_token
        self.pkeep = pkeep
        self.device = device

        self.pretrained_model.eval()



    @torch.no_grad()
    def codebook_encoding(self, img_batch):
        """ Encode an image batch using the pretrained VQVAE model.

        Parameters
        ----------
        img_batch : torch.Tensor
            Image batch.

        Returns
        -------
        torch.Tensor
            Encoded image batch.
        """
        e = self.pretrained_model.encoder(img_batch)
        _, _, _, _, codes = self.pretrained_model.vq_layer(e)   # (B, H, W)
        B = codes.shape[0]
        codes = codes.view(B, -1)  # (B, H*W)
        return codes
    


    def forward(self, x):
        B = x.shape[0]  # batch size
        codes = self.codebook_encoding(x)
        target = codes  # target sequence is the original codes

        sos_tokens = torch.ones(B, 1) * self.sos_token
        sos_tokens = sos_tokens.long().to(device=x.device)

        # replace randomly some tokens with random codes
        mask = torch.bernoulli(self.pkeep * torch.ones_like(codes)).to(dtype=torch.int64).to(device=x.device)
        random_codes = torch.randint_like(codes, self.gpt_model.vocab_size)
        new_codes = mask * codes + (1 - mask) * random_codes
        new_codes = torch.cat((sos_tokens, new_codes), dim=1)

        # gpt model output
        logits = self.gpt_model(new_codes[:, :-1])

        return logits, target


    # Sampling
    def top_k_logits(self, logits, k):
        """Returns the logits data with all values below the top-k set to -infinity.

        Parameters
        ----------
        logits : torch.Tensor
            Logits data.
        k : int
            Number of top values to keep.
        
        Returns
        -------
        torch.Tensor
            Logits data with all values below the top-k set to -infinity.
        """
        values, _ = torch.topk(logits, k)
        out = logits.clone()
        out[out < values[:, [-1]]] = -float('inf')
        return out
    


    @torch.no_grad()
    def sample(self, sos_token, max_len, temperature=1.0, top_k=100):
        """Sample a sequence of codes.

        Parameters
        ----------
        sos_token : torch.Tensor
            Start of sentence token.
        max_len : int
            Maximum length of the sequence.
        temperature : float
            Temperature, higher values increase diversity.
        top_k : int
            Number of top values to keep.

        Returns
        -------
        torch.Tensor
            Sampled sequence of codes.
        """
        self.gpt_model.eval()
        idx = torch.tensor(sos_token).long().to(self.device)
        for i in tqdm(range(max_len)):
            logits = self.gpt_model(idx)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
            
            probs = F.softmax(logits, dim=-1)

            next_idx = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, next_idx), dim=1)
        
        # remove sos token
        idx = idx[:, 1:]

        # reshape to (B, H, W)
        size = torch.sqrt(torch.Tensor([max_len])).to(torch.int64).item()
        idx = idx.view(-1, size, size)
        return idx

    

    # Save model
    def save_model(self, save_folder):
        """Save the model in the specified folder."""
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        param_file = os.path.join(save_folder, 'transformer_parameters.pkl')
        parameters = [self.sos_token, 
                      self.pkeep]
        with open(param_file, 'wb') as f:
            pickle.dump(parameters, f)

        self.gpt_model.save_model(save_folder)


    # Load model
    @staticmethod
    def load_model(save_folder, pretrained_model):
        """Load the model from the specified folder, using the pretrained model."""
        param_file = os.path.join(save_folder, 'transformer_parameters.pkl')
        with open(param_file, 'rb') as f:
            parameters = pickle.load(f)
        gpt_model = GPT.load_model(save_folder)
        model = LatentTransformer(pretrained_model=pretrained_model,
                                  gpt_model=gpt_model,
                                  sos_token=parameters[0],
                                  pkeep=parameters[1])
        return model

        


            

