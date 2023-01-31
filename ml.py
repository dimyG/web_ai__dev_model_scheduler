#%%
import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import os
from pathlib import Path
from dotenv import load_dotenv
import matplotlib.pyplot as plt
#%%

src_path = Path('.')
env_path = src_path / '.env'
load_dotenv(dotenv_path=env_path)  # load environment variables from .env file into the os.environ object.
huggingface_access_token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN")


def img_from_prompt(
        prompt: str, seed: int = 23, guidance_scale: float = 7.5, num_inference_steps: int = 2000) -> torch.Tensor:

    model1 = "CompVis/stable-diffusion-v1-4"
    model2 = "stabilityai/stable-diffusion-2"
    revision = "fp16"  # loading the weights from the float16 precision branch (instead of float32)
    torch_dtype = torch.float16  # telling diffusers to expect the weights to be in float16 precision
    device = "cuda" if torch.cuda.is_available() else "cpu"

    scheduler = EulerDiscreteScheduler.from_pretrained(model2, subfolder="scheduler")

    # Load the model
    pipe = StableDiffusionPipeline.from_pretrained(
        model2,
        scheduler=scheduler,
        revision=revision,
        torch_dtype=torch_dtype,  # telling diffusers to expect the weights to be in float16 precision
        use_auth_token=huggingface_access_token
    )

    pipe.to(device)
    if device == "cuda":
        # gc.collect()
        torch.cuda.empty_cache()
        pipe.enable_attention_slicing()

    # Generate the image
    generator = torch.Generator(device).manual_seed(seed)
    image = pipe(
        prompt=prompt, height=256, width=256, guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps, generator=generator
    ).images[0]

    return image


print(torch.cuda.is_available())

image = img_from_prompt("An astronaut on a horse.")
print(type(image))
# display(image)
plt.imshow(image)
plt.axis('off')
plt.show()
#%%

