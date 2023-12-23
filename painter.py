import numpy as np
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Painter:
    def __init__(self):
        self.vae = AutoencoderKL.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder='vae').to(device)
        self.tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
        self.text_encoder = CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14').to(device)
        self.unet = UNet2DConditionModel.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder='unet').to(device)
        self.scheduler = DDIMScheduler.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder='scheduler')

        self.latents = torch.randn((1, 4, 512 // 8, 512 // 8)).to(device)
        self.scheduler.set_timesteps(30)
        self.latents = self.latents * self.scheduler.init_noise_sigma

    def encode_text(self, text):
        text_input = self.tokenizer([text], padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(device))[0]
        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer([''], padding='max_length', max_length=max_length, return_tensors='pt')
        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(device))[0]
        self.text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    def latents_to_image(self):
        with torch.no_grad():
            image = self.vae.decode(self.latents / .18215).sample
        image = (image / 2 + 0.5)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()[0]
        image = (image * 255).clip(0, 255).astype(np.uint8)
        return image

    def paint(self, strokes):
        latent_model_input = torch.cat([self.latents, self.latents])
        