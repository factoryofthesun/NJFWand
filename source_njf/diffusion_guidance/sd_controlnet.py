import torch
import torch.nn.functional as F
from diffusers import IFPipeline, UNet2DConditionModel
from typing import Optional, Union, Tuple, List
from contextlib import contextmanager

from diffusers import (
    AutoPipelineForText2Image,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
from diffusers.utils import load_image
from diffusers.image_processor import PipelineImageInput
import torch
from dataclasses import dataclass
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel

@dataclass
class ControlConfig:
    """ Config for controlnet SDS """
    # Timestep sampling parameters
    min_step_percent: float = 0.02
    max_step_percent: float = 0.98
    # Default conditioning to depth
    conditioning = "depth"
    # Use img loss from HiFA
    use_img_loss: bool = False
    # CFG guidance scale
    guidance_scale: float = 5 # 5.
    # Whether or not to use half precision weights
    half_precision_weights: bool = True
    # Whether or not to use inpainting
    inpainting: bool = False

class ControlSDS:
    def __init__(self, cfg: ControlConfig = None, device=None, **kwargs) -> None:
        # Set device (cuda or cpu)
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # If no config passed, use default config
        if cfg is None:
            cfg = ControlConfig()
        self.cfg = cfg

        # If kwargs passed, then we use it to set the config params
        for k, v in kwargs.items():
            setattr(self.cfg, k, v)

        # Load model
        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )
        self.load_model()

    def load_model(self):
        if self.cfg.conditioning == "canny":
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-canny",
                torch_dtype=self.weights_dtype
            ).eval()
        elif self.cfg.conditioning == "depth":
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-depth",
                torch_dtype=self.weights_dtype
            ).eval()
        elif self.cfg.conditioning == "normal":
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-normal",
                torch_dtype=self.weights_dtype
            ).eval()

        # NOTE: Pipe on CPU because of memory issues
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=self.weights_dtype
        ).to(self.device)
        self.pipe.vae = self.pipe.vae.eval()
        self.pipe.unet = self.pipe.unet.eval()
        self.pipe.text_encoder = self.pipe.text_encoder.eval()
        self.unet = self.pipe.unet.eval()

        for p in self.unet.parameters():
            p.requires_grad_(False)

        self.scheduler = self.pipe.scheduler

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * self.cfg.min_step_percent)
        self.max_step = int(self.num_train_timesteps * self.cfg.max_step_percent)

        self.alphas: torch.FloatTensor = self.scheduler.alphas_cumprod.to(
            self.device
        )

        self.grad_clip_val: Optional[float] = None

    def encode_prompt(self, prompt: str, do_cfg = True, negative_prompt: str = None):
        return self.pipe.encode_prompt(
            prompt,
            num_images_per_prompt=1,
            device=self.device,
            do_classifier_free_guidance=do_cfg,
            negative_prompt=negative_prompt,
        )

    @torch.cuda.amp.autocast(enabled=False)
    def encode_latent_images(
        self, imgs # Float[Tensor, "B 3 512 512"]
    ): #-> Float[Tensor, "B 4 64 64"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        # imgs = imgs.float()
        # self.pipe.vae.to(dtype=torch.float32)

        # NOTE: VAE must be in float32 mode -- otherwise will overflow
        posterior = self.pipe.vae.encode(imgs.half()).latent_dist
        latents = posterior.sample() * self.pipe.vae.config.scaling_factor

        # self.pipe.vae.to(dtype=torch.float16)

        return latents.to(input_dtype)

    def decode_latents(
        self,
        latents, #: Float[Tensor, "B 4 H W"],
        latent_height: int = 64, # 128
        latent_width: int = 64, # 128 # TODO: change back to 128 when fixing hight/width/orignal size
    ): # -> Float[Tensor, "B 3 1024 1024"]:
        input_dtype = latents.dtype
        latents = F.interpolate(
            latents, (latent_height, latent_width), mode="bilinear", align_corners=False
        )

        latents = 1 / self.pipe.vae.config.scaling_factor * latents
        image = self.pipe.vae.decode(latents.to(self.weights_dtype)).sample

        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    def classifier_free_guidance(self, noise_pred, guidance_scale, output_channels=3, split=True):
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

        if split:
            noise_pred_uncond, _ = noise_pred_uncond.split(output_channels, dim=1)
            noise_pred_text, _ = noise_pred_text.split(output_channels, dim=1)
        noise_pred = noise_pred_text + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
        return noise_pred

    @contextmanager
    def disable_unet_class_embedding(self, unet: UNet2DConditionModel):
        class_embedding = unet.class_embedding
        try:
            unet.class_embedding = None
            yield unet
        finally:
            unet.class_embedding = class_embedding

    def compute_grad_base(
        self,
        z_0: torch.FloatTensor, # B, 3, H, W
        text_embeddings: torch.FloatTensor, # 2B, 77, 2048
        control_image: Optional[torch.FloatTensor] = None, # B, 3, H, W
    ):
        B = z_0.shape[0]

        with torch.no_grad():
            # timestep ~ U(min_step, max_step)
            t = torch.randint(
                self.min_step,
                self.max_step + 1,
                [B],
                dtype=torch.long,
                device=self.device,
            )

            # add noise
            noise = torch.randn_like(z_0)
            z_t = self.scheduler.add_noise(z_0, noise, t)

            # Control Net: classifier free guidance
            latent_model_input = torch.cat([z_t] * 2, dim=0).half()
            control_model_input = latent_model_input

            down_block_res_samples, mid_block_res_sample = self.pipe.controlnet(
                    control_model_input,
                    torch.cat([t] * 2),
                    encoder_hidden_states=text_embeddings.repeat_interleave(B, dim=0),
                    controlnet_cond=torch.cat([control_image] * 2, dim=0).half(),
                    return_dict=False,
            )

            timestep_cond = None
            # TODO: MAYBE
            # if self.unet.config.time_cond_proj_dim is not None:
            #     guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            #     timestep_cond = self.get_guidance_scale_embedding(
            #         guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            #     ).to(device=device, dtype=latents.dtype)

            # predict noise
            # add_time_ids = list((z_0.shape[2], z_0.shape[3]) + (0,0) + (z_0.shape[2], z_0.shape[3]))
            # add_time_ids = torch.tensor([add_time_ids], dtype=latent_model_input.dtype, device=self.device)

            with self.disable_unet_class_embedding(self.unet) as unet:
                noise_pred = unet(
                    latent_model_input.half(),
                    torch.cat([t] * 2),
                    encoder_hidden_states=text_embeddings.repeat_interleave(B, dim=0).half(),
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=None,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    # added_cond_kwargs={'text_embeds': text_embeddings.mean(dim=1).repeat_interleave(B, dim=0),
                    #                    'time_ids': torch.cat([add_time_ids] * 2).repeat(B, 1)},
                    return_dict=False,
                )[0]

        noise_pred = self.classifier_free_guidance(noise_pred, self.cfg.guidance_scale, output_channels=4, split=False)
        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)

        alpha = (self.alphas[t] ** 0.5).view(-1, 1, 1, 1)
        sigma = ((1 - self.alphas[t]) ** 0.5).view(-1, 1, 1, 1)
        latents_denoised = (z_t - sigma * noise_pred) / alpha
        image_denoised = self.decode_latents(latents_denoised)

        grad = w * (noise_pred - noise)

        # image-space SDS proposed in HiFA: https://hifa-team.github.io/HiFA-site/
        if self.cfg.use_img_loss:
            grad_img = w * (image - image_denoised) * alpha / sigma
        else:
            grad_img = None

        return grad, grad_img

    # TODO: Run full denoising with controlnet and export the render
    def check_inputs(
        self,
        prompt,
        prompt_2,
        height,
        width,
        negative_prompt=None,
        negative_prompt_2=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif negative_prompt_2 is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_2`: {negative_prompt_2} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."
            )

        if negative_prompt_embeds is not None and negative_pooled_prompt_embeds is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_pooled_prompt_embeds` also have to be passed. Make sure to generate `negative_pooled_prompt_embeds` from the same text encoder that was used to generate `negative_prompt_embeds`."
            )

    def check_conditions(
        self,
        prompt,
        control_image,
        controlnet_conditioning_scale,
        control_guidance_start,
        control_guidance_end,
    ):
        # controlnet checks
        if not isinstance(control_guidance_start, (tuple, list)):
            control_guidance_start = [control_guidance_start]

        if not isinstance(control_guidance_end, (tuple, list)):
            control_guidance_end = [control_guidance_end]

        if len(control_guidance_start) != len(control_guidance_end):
            raise ValueError(
                f"`control_guidance_start` has {len(control_guidance_start)} elements, but `control_guidance_end` has {len(control_guidance_end)} elements. Make sure to provide the same number of elements to each list."
            )

        if isinstance(self.pipe.controlnet, MultiControlNetModel):
            if len(control_guidance_start) != len(self.pipe.controlnet.nets):
                raise ValueError(
                    f"`control_guidance_start`: {control_guidance_start} has {len(control_guidance_start)} elements but there are {len(self.pipe.controlnet.nets)} controlnets available. Make sure to provide {len(self.pipe.controlnet.nets)}."
                )

        for start, end in zip(control_guidance_start, control_guidance_end):
            if start >= end:
                raise ValueError(
                    f"control guidance start: {start} cannot be larger or equal to control guidance end: {end}."
                )
            if start < 0.0:
                raise ValueError(f"control guidance start: {start} can't be smaller than 0.")
            if end > 1.0:
                raise ValueError(f"control guidance end: {end} can't be larger than 1.0.")

        # Check controlnet `image`
        is_compiled = hasattr(F, "scaled_dot_product_attention") and isinstance(
            self.pipe.controlnet, torch._dynamo.eval_frame.OptimizedModule
        )
        if (
            isinstance(self.pipe.controlnet, ControlNetModel)
            or is_compiled
            and isinstance(self.pipe.controlnet._orig_mod, ControlNetModel)
        ):
            self.check_image(control_image, prompt)
        elif (
            isinstance(self.pipe.controlnet, MultiControlNetModel)
            or is_compiled
            and isinstance(self.pipe.controlnet._orig_mod, MultiControlNetModel)
        ):
            if not isinstance(control_image, list):
                raise TypeError("For multiple controlnets: `control_image` must be type `list`")

            # When `image` is a nested list:
            # (e.g. [[canny_image_1, pose_image_1], [canny_image_2, pose_image_2]])
            elif any(isinstance(i, list) for i in control_image):
                raise ValueError("A single batch of multiple conditionings are supported at the moment.")
            elif len(control_image) != len(self.pipe.controlnet.nets):
                raise ValueError(
                    f"For multiple controlnets: `image` must have the same length as the number of controlnets, but got {len(control_image)} images and {len(self.pipe.controlnet.nets)} ControlNets."
                )

            for image_ in control_image:
                self.check_image(image_, prompt)
        else:
            assert False

        # Check `controlnet_conditioning_scale`
        if (
            isinstance(self.pipe.controlnet, ControlNetModel)
            or is_compiled
            and isinstance(self.pipe.controlnet._orig_mod, ControlNetModel)
        ):
            if not isinstance(controlnet_conditioning_scale, float):
                raise TypeError("For single controlnet: `controlnet_conditioning_scale` must be type `float`.")
        elif (
            isinstance(self.pipe.controlnet, MultiControlNetModel)
            or is_compiled
            and isinstance(self.pipe.controlnet._orig_mod, MultiControlNetModel)
        ):
            if isinstance(controlnet_conditioning_scale, list):
                if any(isinstance(i, list) for i in controlnet_conditioning_scale):
                    raise ValueError("A single batch of multiple conditionings are supported at the moment.")
            elif isinstance(controlnet_conditioning_scale, list) and len(controlnet_conditioning_scale) != len(
                self.pipe.controlnet.nets
            ):
                raise ValueError(
                    "For multiple controlnets: When `controlnet_conditioning_scale` is specified as `list`, it must have"
                    " the same length as the number of controlnets"
                )
        else:
            assert False

    def check_image(self, image, prompt):
        import PIL
        import numpy as np

        image_is_pil = isinstance(image, PIL.Image.Image)
        image_is_tensor = isinstance(image, torch.Tensor)
        image_is_np = isinstance(image, np.ndarray)
        image_is_pil_list = isinstance(image, list) and isinstance(image[0], PIL.Image.Image)
        image_is_tensor_list = isinstance(image, list) and isinstance(image[0], torch.Tensor)
        image_is_np_list = isinstance(image, list) and isinstance(image[0], np.ndarray)

        if (
            not image_is_pil
            and not image_is_tensor
            and not image_is_np
            and not image_is_pil_list
            and not image_is_tensor_list
            and not image_is_np_list
        ):
            raise TypeError(
                f"image must be passed and be one of PIL image, numpy array, torch tensor, list of PIL images, list of numpy arrays or list of torch tensors, but is {type(image)}"
            )

        if image_is_pil:
            image_batch_size = 1
        else:
            image_batch_size = len(image)

        if prompt is not None and isinstance(prompt, str):
            prompt_batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            prompt_batch_size = len(prompt)

        if image_batch_size != 1 and image_batch_size != prompt_batch_size:
            raise ValueError(
                f"If image batch size is not 1, image batch size must be same as prompt batch size. image batch size: {image_batch_size}, prompt batch size: {prompt_batch_size}"
            )

    def upcast_vae(self):
        from diffusers.models.attention_processor import (
            AttnProcessor2_0,
            LoRAAttnProcessor2_0,
            LoRAXFormersAttnProcessor,
            XFormersAttnProcessor,
        )

        dtype = self.pipe.vae.dtype
        self.pipe.vae.to(dtype=torch.float32)
        use_torch_2_0_or_xformers = isinstance(
            self.pipe.vae.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
                LoRAXFormersAttnProcessor,
                LoRAAttnProcessor2_0,
            ),
        )
        # if xformers or torch_2_0 is used attention block does not need
        # to be in float32 which can save lots of memory
        if use_torch_2_0_or_xformers:
            self.pipe.vae.post_quant_conv.to(dtype)
            self.pipe.vae.decoder.conv_in.to(dtype)
            self.pipe.vae.decoder.mid_block.to(dtype)

    def full_denoise(
        self,
        noised_latents,
        prompt,
        prompt_2: Optional[Union[str, List[str]]] = None,
        control_image: PipelineImageInput = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        num_images_per_prompt: Optional[int] = 1,
        output_type: Optional[str] = "pil",
        original_size: Optional[Tuple[int, int]] = None,
        target_size: Optional[Tuple[int, int]] = None,
        controlnet_conditioning_scale=1.0,
        guess_mode: bool = False,
        control_guidance_start: float = 0.0,
        control_guidance_end: float = 1.0,
    ):
        controlnet = self.pipe.controlnet

        # 0. Default height and width is h/w of input noised latent
        height, width = noised_latents.shape[2], noised_latents.shape[3]
        original_size = (height, width)
        target_size = (height, width)

        # 0.1 align format for control guidance
        mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
        control_guidance_start, control_guidance_end = (
            mult * [control_guidance_start],
            mult * [control_guidance_end],
        )

        if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
        )

        self.check_conditions(
            prompt,
            control_image,
            controlnet_conditioning_scale,
            control_guidance_start,
            control_guidance_end,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = self.cfg.guidance_scale > 1.0

        # 3. Encode input prompt
        (
            prompt_embeds,
            negative_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            do_cfg=do_classifier_free_guidance,
        )

        # 4. Prepare timesteps
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=self.device)

        timesteps = self.pipe.scheduler.timesteps

        # 5. Prepare latent variables
        latents = noised_latents

        # 7.2 Prepare control images
        control_image = torch.cat([control_image] * 2, dim=0)

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        prompt_embeds = prompt_embeds.to(self.device)

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        from tqdm.contrib import tenumerate

        for i, t in tenumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # ----------- ControlNet

            # expand the latents if we are doing classifier free guidance
            latent_model_input_controlnet = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

            # concat latents, mask, masked_image_latents in the channel dimension
            latent_model_input_controlnet = self.scheduler.scale_model_input(latent_model_input_controlnet, t)

            # controlnet(s) inference
            if guess_mode and do_classifier_free_guidance:
                # Infer ControlNet only for the conditional batch.
                control_model_input = latents
                control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
            else:
                control_model_input = latent_model_input_controlnet
                controlnet_prompt_embeds = prompt_embeds

            down_block_res_samples, mid_block_res_sample = self.pipe.controlnet(
                control_model_input.half(),
                t,
                encoder_hidden_states=controlnet_prompt_embeds,
                controlnet_cond=control_image.half(),
                guess_mode=guess_mode,
                return_dict=False,
            )

            noise_pred = self.pipe.unet(
                latent_model_input.half(),
                t,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
                down_block_additional_residuals=down_block_res_samples,  # controlnet
                mid_block_additional_residual=mid_block_res_sample,  # controlnet
            )[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # make sure the VAE is in float32 mode, as it overflows in float16
        needs_upcasting = self.pipe.vae.dtype == torch.float16 and self.pipe.vae.config.force_upcast

        if needs_upcasting:
            self.upcast_vae()
            latents = latents.to(next(iter(self.pipe.vae.post_quant_conv.parameters())).dtype)

        image = self.pipe.vae.decode(latents / self.pipe.vae.config.scaling_factor, return_dict=False)[0]

        # cast back to fp16 if needed
        if needs_upcasting:
            self.pipe.vae.to(dtype=torch.float16)

        from diffusers.image_processor import VaeImageProcessor
        vae_scale_factor = 2 ** (len(self.pipe.vae.config.block_out_channels) - 1)
        image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
        image = image_processor.postprocess(image.detach(), output_type="pil")

        return image

    # def full_denoise(
    #     self,
    #     noise_latent, # Initial noised latents
    #     text_prompt: torch.FloatTensor, # 2B, 77, 2048
    #     control_image, # PIL Image
    # ):
    #     self.pipe.enable_xformers_memory_efficient_attention()
    #     image = self.pipe(latents = noise_latent, prompt = text_prompt, image=control_image,
    #                       generator=generator).images[0]
    #     return image

    def compute_grad_base(
        self,
        z_0: torch.FloatTensor, # B, 3, H, W
        text_embeddings: torch.FloatTensor, # 2B, 77, 2048
        control_image: Optional[torch.FloatTensor] = None, # B, 3, H, W
    ):
        B = z_0.shape[0]

        with torch.no_grad():
            # timestep ~ U(min_step, max_step)
            t = torch.randint(
                self.min_step,
                self.max_step + 1,
                [B],
                dtype=torch.long,
                device=self.device,
            )

            # add noise
            noise = torch.randn_like(z_0)
            z_t = self.scheduler.add_noise(z_0, noise, t)

            # Control Net: classifier free guidance
            latent_model_input = torch.cat([z_t] * 2, dim=0).half()
            control_model_input = latent_model_input

            down_block_res_samples, mid_block_res_sample = self.pipe.controlnet(
                    control_model_input,
                    torch.cat([t] * 2),
                    encoder_hidden_states=text_embeddings.repeat_interleave(B, dim=0),
                    controlnet_cond=torch.cat([control_image] * 2, dim=0).half().half(),
                    return_dict=False,
            )

            timestep_cond = None
            # TODO: MAYBE
            # if self.unet.config.time_cond_proj_dim is not None:
            #     guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            #     timestep_cond = self.get_guidance_scale_embedding(
            #         guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            #     ).to(device=device, dtype=latents.dtype)

            # predict noise
            with self.disable_unet_class_embedding(self.unet) as unet:
                noise_pred = unet(
                    latent_model_input,
                    torch.cat([t] * 2),
                    encoder_hidden_states=text_embeddings.repeat_interleave(B, dim=0),
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=None,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    added_cond_kwargs=None,
                    return_dict=False,
                )[0]

        noise_pred = self.classifier_free_guidance(noise_pred, self.cfg.guidance_scale, output_channels=4, split=False)
        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)

        alpha = (self.alphas[t] ** 0.5).view(-1, 1, 1, 1)
        sigma = ((1 - self.alphas[t]) ** 0.5).view(-1, 1, 1, 1)
        latents_denoised = (z_t - sigma * noise_pred) / alpha
        image_denoised = self.decode_latents(latents_denoised)

        grad = w * (noise_pred - noise)

        # image-space SDS proposed in HiFA: https://hifa-team.github.io/HiFA-site/
        if self.cfg.use_img_loss:
            grad_img = w * (image - image_denoised) * alpha / sigma
        else:
            grad_img = None

        return grad, grad_img

    def __call__(
        self,
        rgb_img: torch.FloatTensor, # B 3 H W
        prompt_embeds,
        negative_embeds,
        control_image: torch.FloatTensor, # B 3 H W
        **kwargs,
    ):
        batch_size = rgb_img.shape[0]
        # rgb_img = rgb_img * 2.0 - 1.0  # scale to [-1, 1] to match the diffusion range

        # encode RGB image to latent space
        rgb_BCHW_512 = F.interpolate(rgb_img, (512, 512), mode="bilinear", align_corners=False)
        latents = self.encode_latent_images(rgb_BCHW_512)
        z_0 = latents

        # Set prompt embeddings
        # NOTE: Negative embed here is always just the no-condition embedding!
        self.text_embeddings = torch.cat([negative_embeds, prompt_embeds])

        grad, grad_img = self.compute_grad_base(
            z_0,
            text_embeddings=self.text_embeddings,
            control_image=control_image,
        )

        if self.cfg.use_img_loss:
            grad_img = torch.nan_to_num(grad_img)
            if self.grad_clip_val is not None:
                grad_img = grad_img.clamp(-self.grad_clip_val, self.grad_clip_val)
            target = (rgb_BCHW_512 - grad_img).detach()
            loss = (
                0.5 * F.mse_loss(rgb_BCHW_512, target, reduction="sum") / batch_size
            )
        else:
            grad = torch.nan_to_num(grad)
            if self.grad_clip_val is not None:
                grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)
            target = (z_0 - grad).detach()
            loss = 0.5 * F.mse_loss(z_0, target, reduction="sum") / batch_size

        return {
            "loss_sds": loss,
            "target": target,
            "grad_norm": grad.norm(),
            "min_step": self.min_step,
            "max_step": self.max_step,
        }


    def update_step(self, min_step_percent: float, max_step_percent: float):
        self.set_min_max_steps(
            min_step_percent=min_step_percent,
            max_step_percent=max_step_percent,
        )