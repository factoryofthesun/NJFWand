import torch
import torch.nn.functional as F
from diffusers import IFPipeline, IFSuperResolutionPipeline
from typing import Optional
from dataclasses import dataclass

@dataclass
class DiffusionConfig:
    """ Config for diffusion sds """
    # Stage I model name
    stage_I_model: str = "DeepFloyd/IF-I-XL-v1.0"
    # Stage II model name
    stage_II_model: str = "DeepFloyd/IF-II-L-v1.0"
    # Timestep sampling parameters
    min_step_percent: float = 0.02
    max_step_percent: float = 0.98
    # CFG guidance scale
    guidance_scale: float = 20 #20.0
    # Whether or not to use half precision weights
    half_precision_weights: bool = True
    # Whether or not to use inpainting
    inpainting: bool = False
    # Whether or not to use cascaded sds
    cascaded: bool = True

class DeepFloydCascaded:
    def __init__(self, cfg: DiffusionConfig=None, device=None, **kwargs):
        # Set device (cuda or cpu)
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # If no config passed, use default config
        if cfg is None:
            cfg = DiffusionConfig()
        self.cfg = cfg

        # If kwargs passed, then we use it to set the config params
        for k, v in kwargs.items():
            setattr(self.cfg, k, v)

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        # Load Huggingface Diffusors pipelines
        self.load_stage_I() # Stage I
        if self.cfg.cascaded:
            self.load_stage_II() # Stage II

        self.grad_clip_val: Optional[float] = None

    def load_stage_I(self):
        self.stage_I_pipe = IFPipeline.from_pretrained(
            self.cfg.stage_I_model,
            safety_checker=None,
            watermarker=None,
            feature_extractor=None,
            requires_safety_checker=False,
            variant="fp16" if self.cfg.half_precision_weights else None,
            torch_dtype=self.weights_dtype,
        ).to(self.device)
        self.stage_I_unet = self.stage_I_pipe.unet.eval()
        for p in self.stage_I_unet.parameters():
            p.requires_grad_(False)
        self.stage_I_scheduler = self.stage_I_pipe.scheduler
        self.stage_I_num_train_timesteps = self.stage_I_scheduler.config.num_train_timesteps
        self.stage_I_min_step = int(self.stage_I_num_train_timesteps * self.cfg.min_step_percent)
        self.stage_I_max_step = int(self.stage_I_num_train_timesteps * self.cfg.max_step_percent)
        self.stage_I_alphas: torch.FloatTensor = self.stage_I_scheduler.alphas_cumprod.to(self.device)

    def load_stage_II(self):
        self.stage_II_pipe = IFSuperResolutionPipeline.from_pretrained(
            self.cfg.stage_II_model,
            text_encoder=None,
            safety_checker=None,
            watermarker=None,
            feature_extractor=None,
            requires_safety_checker=False,
            variant="fp16" if self.cfg.half_precision_weights else None,
            torch_dtype=self.weights_dtype,
        ).to(self.device)
        self.stage_II_unet = self.stage_II_pipe.unet.eval()
        for p in self.stage_II_unet.parameters():
            p.requires_grad_(False)
        self.stage_II_scheduler = self.stage_II_pipe.scheduler
        self.stage_II_num_train_timesteps = self.stage_II_scheduler.config.num_train_timesteps
        self.stage_II_min_step = int(self.stage_II_num_train_timesteps * self.cfg.min_step_percent)
        self.stage_II_max_step = int(self.stage_II_num_train_timesteps * self.cfg.max_step_percent)
        self.stage_II_alphas: torch.FloatTensor = self.stage_II_scheduler.alphas_cumprod.to(self.device)

    def encode_prompt(self, prompt: str, negative_prompt: str = None):
        # NOTE: Classifier free guidance defaults to True in encode_prompt
        prompt_embeds, negative_embeds = self.stage_I_pipe.encode_prompt(prompt, negative_prompt=negative_prompt)
        if self.cfg.guidance_scale > 1.0:
            self.prompt_embeds = torch.cat([negative_embeds, prompt_embeds])
        else:
            self.prompt_embeds = prompt_embeds
        return prompt_embeds, negative_embeds

    def mask_images(self, background, selection, mask_image):
        mask_image = mask_image.unsqueeze(1).repeat_interleave(3, dim=1)
        return (1 - mask_image) * background + mask_image * selection

    def classifier_free_guidance(self, noise_pred, output_channels=3, keep_variance=False):
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred_uncond, _ = noise_pred_uncond.split(output_channels, dim=1)
        noise_pred_text, predicted_variance = noise_pred_text.split(output_channels, dim=1)
        noise_pred = noise_pred_text + self.cfg.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        if keep_variance:
            noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)

        return noise_pred

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        latents: torch.FloatTensor,
        t: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        stage: str = "I",
        class_labels: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        input_dtype = latents.dtype
        if stage == "I":
            return self.stage_I_unet(
                latents.to(self.weights_dtype),
                t.to(self.weights_dtype),
                encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            ).sample.to(input_dtype)
        elif stage == "II":
            return self.stage_II_unet(
                latents.to(self.weights_dtype),
                t.to(self.weights_dtype),
                encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
                class_labels=class_labels.to(self.weights_dtype),
            ).sample.to(input_dtype)
        else:
            raise ValueError(f"Invalid stage {stage}")

    def __call__(
        self,
        rgb_img: torch.FloatTensor, # Float[Tensor, "B C H W"]
        mask_img: torch.FloatTensor = None, # Float[Tensor, "B 1 H W"]
        prompt_embeds: Optional[str] = None,
        negative_embeds: Optional[str] = None,
        stage: str = "I",
        tratio = None,
        **kwargs,
    ):
        batch_size = rgb_img.shape[0]

        # If prompt embeddings are passed, overwrite existing prompt embeddings
        if prompt_embeds is not None and negative_embeds is not None:
            self.prompt_embeds = torch.cat([negative_embeds, prompt_embeds])

        rgb_img = rgb_img * 2.0 - 1.0  # scale to [-1, 1] to match the diffusion range

        if stage == "I":
            z_0 = F.interpolate(rgb_img, (64, 64), mode="bilinear", align_corners=False)

            # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
            t = torch.randint(
                self.stage_I_min_step,
                self.stage_I_max_step + 1,
                [batch_size],
                dtype=torch.long,
                device=self.device,
            )

            # If given time step ratio, then use that instead
            if tratio is not None:
                t = torch.tensor([int(self.stage_I_min_step + tratio * (self.stage_I_max_step + 1 - self.stage_I_min_step))] * batch_size, device=self.device)

            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                # add noise
                noise = torch.randn_like(z_0)
                z_t = self.stage_I_scheduler.add_noise(z_0, noise, t)

                # pred noise
                model_input = torch.cat([z_t] * 2, dim=0)
                noise_pred = self.forward_unet(
                    model_input,
                    torch.cat([t] * 2),
                    encoder_hidden_states=self.prompt_embeds.repeat_interleave(batch_size, dim=0),
                    stage="I",
                )  # (B, 6, 64, 64)

            # Perform classifier free guidance
            noise_pred = self.classifier_free_guidance(noise_pred)

            # w(t), sigma_t^2
            w_t = (1 - self.stage_I_alphas[t]).view(-1, 1, 1, 1)

            grad = w_t * (noise_pred - noise)
            grad = torch.nan_to_num(grad)
            # clip grad for stable training (per threestudio)
            if self.grad_clip_val is not None:
                grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

            # Use reparameterization trick to avoid backproping gradient
            target = (z_0 - grad).detach()
            # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
            loss_sds = 0.5 * F.mse_loss(z_0, target, reduction="sum") / batch_size

        elif stage == "II":
            z_0 = F.interpolate(rgb_img, (64, 64), mode="bilinear", align_corners=False)
            # z_0 = F.interpolate(self.img_tensor.unsqueeze(0), (64, 64), mode="bilinear", align_corners=False)
            x_0 = F.interpolate(rgb_img, (256, 256), mode="bilinear", align_corners=False)

            # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
            timesteps = torch.randint(
                self.stage_II_min_step,
                self.stage_II_max_step + 1,
                [batch_size*2],
                dtype=torch.long,
                device=self.device,
            )
            t, s = timesteps.chunk(2, dim=0)

            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                # add noise
                upscaled_z_0 = F.interpolate(z_0, (256, 256), mode="bilinear", align_corners=False)
                noise = torch.randn_like(x_0.repeat(2, 1, 1, 1))
                noise_s, noise_t = noise.chunk(2, dim=0)
                upscaled_z_s = self.stage_II_scheduler.add_noise(upscaled_z_0, noise_s, s)
                x_t = self.stage_II_scheduler.add_noise(x_0, noise_t, t)
                latents_noisy = torch.cat([x_t, upscaled_z_s], dim=1)

                # pred noise
                model_input = torch.cat([latents_noisy] * 2, dim=0)
                noise_pred = self.forward_unet(
                    model_input,
                    torch.cat([t] * 2),
                    encoder_hidden_states=self.prompt_embeds.repeat_interleave(batch_size, dim=0),
                    class_labels=torch.cat([s] * 2), # this argument allows the unet to take in s
                    stage="II",
                )  # (B, 12, 256, 256)

            # Perform classifier free guidance
            noise_pred = self.classifier_free_guidance(noise_pred)

            # w(t), sigma_t^2
            w_t = (1 - self.stage_II_alphas[t]).view(-1, 1, 1, 1)

            grad = w_t * (noise_pred - noise_t)
            grad = torch.nan_to_num(grad)
            # clip grad for stable training (per threestudio)
            if self.grad_clip_val is not None:
                grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

            # Use reparameterization trick to avoid backproping gradient
            target = (x_0 - grad).detach()
            # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
            loss_sds = 0.5 * F.mse_loss(x_0, target, reduction="sum") / batch_size

        return {
            "loss_sds": loss_sds,
            "grad": grad,
            "grad_norm": grad.norm(),
            "target": target,
        }

    def full_denoise(
        self,
        rgb_img: torch.FloatTensor, # Float[Tensor, "B C H W"]
        prompt_embeds,
        negative_embeds,
        stage: str = "I",
        tratio = None,
        **kwargs,
    ):

        if stage == "I":
            timesteps = 100

            batch_size = rgb_img.shape[0]
            rgb_img = rgb_img * 2.0 - 1.0  # scale to [-1, 1] to match the diffusion range

            self.prompt_embeds = torch.cat([negative_embeds, prompt_embeds])

            z_0 = F.interpolate(rgb_img, (64, 64), mode="bilinear", align_corners=False)
            noise = torch.randn_like(z_0)
            intermediate_images = self.stage_I_scheduler.add_noise(z_0, noise, torch.tensor(100, device=noise.device))

            from tqdm.contrib import tenumerate

            for i, t in tenumerate(range(timesteps)):
                model_input = (
                    torch.cat([intermediate_images] * 2)
                )

                model_input = self.stage_I_scheduler.scale_model_input(model_input, t)
                noise_pred = self.forward_unet(
                    model_input,
                    torch.cat([torch.tensor([t], device=model_input.device)] * 2 * batch_size),
                    encoder_hidden_states=self.prompt_embeds.repeat_interleave(batch_size, dim=0),
                    stage="I",
                )  # (B, 6, 64, 64)

                # Perform classifier free guidance
                noise_pred = self.classifier_free_guidance(noise_pred, keep_variance=True)

                intermediate_images = self.stage_I_scheduler.step(
                    noise_pred, t, intermediate_images, return_dict=False
                )[0]

            return intermediate_images

        elif stage == "II":
            image = self.stage_II_pipe(
                image=rgb_img,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_embeds,
                output_type="pt",
            ).images

            return image