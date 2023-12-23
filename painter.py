import numpy as np
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from skimage.draw import disk


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Painter:
    def __init__(self):
        self.vae = AutoencoderKL.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder='vae').to(device)
        self.tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
        self.text_encoder = CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14').to(device)
        self.unet = UNet2DConditionModel.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder='unet').to(device)
        self.scheduler = DDIMScheduler.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder='scheduler')

        self.latents = torch.randn((1, 4, 512 // 8, 512 // 8)).to(device)
        self.stroke_map = torch.zeros((512 // 8, 512 // 8)).to(device)
        self.patch_radius = 5
        self.scheduler.set_timesteps(30)
        self.guidance_scale = 7.5
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
        patch = torch.zeros_like(self.stroke_map)
        for stroke in strokes:
            rr, cc = disk((stroke['x'], stroke['y']), self.patch_radius, shape=patch.shape)
            patch[rr, cc] = 1

        estimated_num_steps = self.stroke_map[patch == 1].mean()
        estimated_num_steps = 0 if torch.isnan(estimated_num_steps) else int(estimated_num_steps)
        estimated_num_steps = max(estimated_num_steps, len(self.scheduler.timesteps) - 1)
        t = self.scheduler.timesteps[estimated_num_steps]

        latent_model_input = torch.cat([self.latents, self.latents])
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
        with torch.no_grad():
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=self.text_embeddings).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

        noise_pred = patch[None, None, :, :] * noise_pred
        max_channel = torch.argmax(noise_pred.abs().sum(dim=(0, 2, 3)))
        masked_noise_pred = torch.zeros_like(noise_pred)
        masked_noise_pred[:, max_channel, :, :] = noise_pred[:, max_channel, :, :]

        self.latents = self.scheduler.step(masked_noise_pred, t, self.latents).prev_sample
        self.stroke_map += patch
