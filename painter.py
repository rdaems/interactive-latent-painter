import numpy as np
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import DiffusionPipeline, AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from torchvision.transforms.functional import gaussian_blur, resize
from skimage.draw import disk


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Painter:
    def __init__(self):
        # use self.pipeline.vae_scale_factor!
        self.pipeline = DiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', torch_dtype=torch.float16, use_safetensors=True).to('cuda')
        self.vae = self.pipeline.vae
        self.tokenizer = self.pipeline.tokenizer
        self.text_encoder = self.pipeline.text_encoder
        self.unet = self.pipeline.unet
        # self.scheduler = self.pipeline.scheduler
        # self.vae = AutoencoderKL.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder='vae').to(device)
        # self.tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
        # self.text_encoder = CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14').to(device)
        # self.unet = UNet2DConditionModel.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder='unet').to(device)
        self.scheduler = DDIMScheduler.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder='scheduler')
        self.dtype = self.pipeline.dtype

        self.image_width = 768
        self.image_height = 768
        self.latents = torch.randn((1, 4, self.image_height // 8, self.image_width // 8), dtype=self.dtype).to(device)
        self.stroke_map = torch.zeros((self.image_height // 8, self.image_width // 8), dtype=self.dtype).to(device)
        self.patch_radius = 32
        self.time_threshold = .8
        self.noise_brush = True
        self.num_timesteps = 30
        self.scheduler.set_timesteps(self.num_timesteps)
        self.guidance_scale = 7.5

        self.latents = self.latents * self.scheduler.init_noise_sigma
        self.pred_latents = torch.zeros_like(self.latents)

    def set_brush(self, prompt=None, noise=False):
        if noise:
            self.noise_brush = True
        else:
            self.noise_brush = False
            self.encode_text(prompt)

    def encode_text(self, text):
        text_input = self.tokenizer([text], padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(device))[0]
        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer([''], padding='max_length', max_length=max_length, return_tensors='pt')
        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(device))[0]
        self.text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    def latents_to_image(self, latents=None):
        if latents is None:
            latents = self.pred_latents
        with torch.no_grad():
            image = self.vae.decode(latents / .18215).sample
        image = (image / 2 + 0.5)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()[0]
        image = (image * 255).clip(0, 255).astype(np.uint8)
        return image

    def timemap_to_image(self):
        timemap = self.stroke_map / self.num_timesteps
        timemap = resize(timemap[None], (self.image_height, self.image_width), interpolation=0)[0]
        timemap = timemap.cpu().float().numpy()
        timemap = (timemap * 255).clip(0, 255).astype(np.uint8)
        return timemap

    def paint(self, strokes):
        patch = torch.zeros(len(strokes), self.image_height // 8, self.image_width // 8, dtype=self.dtype).to(device)
        # for stroke in strokes:
        #     rr, cc = disk((stroke['y'] // 8, stroke['x'] // 8), self.patch_radius // 8, shape=patch.shape)
        #     patch[rr, cc] = 1

        for p, stroke in zip(patch, strokes):
            p[stroke['y'] // 8, stroke['x'] // 8] = 1
        sigma = self.patch_radius / 8 / 2
        kernel_size = int(sigma * 4)
        if kernel_size % 2 == 0:
            kernel_size += 1
        patch = gaussian_blur(patch, kernel_size, sigma) * 2 * torch.pi * sigma ** 2
        patch, _ = patch.max(dim=0)

        if patch.sum() == 0:
            return self.latents_to_image()
        if self.noise_brush:
            estimated_num_steps = (self.stroke_map * (patch / patch.sum())).sum()
            estimated_num_steps = 0 if torch.isnan(estimated_num_steps) else int(estimated_num_steps)
            estimated_num_steps = min(estimated_num_steps, len(self.scheduler.timesteps) - 1)
            t = self.scheduler.timesteps[estimated_num_steps]

            noise = torch.randn_like(self.latents)
            noisy_latents = self.scheduler.add_noise(self.latents, noise, t)

            mask = patch[None, None, :, :]
            self.latents = mask * noisy_latents + (1 - mask) * self.latents
            self.pred_latents = mask * self.pred_latents + (1 - mask) * self.pred_latents

            self.stroke_map = torch.clip(self.stroke_map - patch, 0, self.num_timesteps)
        else:
            patch[self.stroke_map > self.time_threshold * self.num_timesteps] = 0
            if patch.sum() == 0:
                return self.latents_to_image()
            estimated_num_steps = (self.stroke_map * (patch / patch.sum())).sum()
            estimated_num_steps = 0 if torch.isnan(estimated_num_steps) else int(estimated_num_steps)
            estimated_num_steps = min(estimated_num_steps, len(self.scheduler.timesteps) - 1)
            t = self.scheduler.timesteps[estimated_num_steps]

            latent_model_input = torch.cat([self.latents, self.latents])
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=self.text_embeddings).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            mask = patch[None, None, :, :]
            # noise_pred_patch = patch[None, None, :, :] * noise_pred
            # max_channel = torch.argmax((noise_pred_patch ** 2).sum(dim=(0, 2, 3)))
            # mask = torch.zeros_like(self.latents)
            # mask[:, max_channel, :, :] = patch[None, :, :]

            step = self.scheduler.step(noise_pred, t, self.latents)
            self.latents = mask * step.prev_sample + (1 - mask) * self.latents
            self.pred_latents = mask * step.pred_original_sample + (1 - mask) * self.pred_latents

            self.stroke_map = torch.clip(self.stroke_map + patch, 0, self.num_timesteps)
        return self.latents_to_image()
