import io
import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from ml import img_from_prompt, load_pipeline, StableDiffusionPipeline
# from huggingface_hub import try_to_load_from_cache, scan_cache_dir

src_path = Path('.')
env_path = src_path / '.env'
load_dotenv(dotenv_path=env_path)  # load environment variables from .env file into the os.environ object.

# We read the defined cache dir from the env file, and then we use the cache_dir parameter to use it.
# Using the cache_dir parameter could be avoided if we set the HUGGINGFACE_HUB_CACHE environment variable on the shell.
# But it doesn't work just by adding it in the env file, so we read it from the env file and then set it manually.
huggingface_hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
debug = os.environ.get("DEBUG")
print("huggingface_hub_cache:", huggingface_hub_cache)
print("debug:", debug)

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model1 = "CompVis/stable-diffusion-v1-4"
model2 = "stabilityai/stable-diffusion-2"
# try_to_load_from_cache
# print(scan_cache_dir())

pipe_initial: StableDiffusionPipeline | None = None

if not debug:
    # in production: load the pipeline in memory on host start, and then reuse it for all images you want to generate.
    # in dev: don't load it on start because it slows down the hot reloading on code changes.
    # Instead, load it on each request to generate_img (which will make it slower, but it's ok for dev)
    pipe_initial = load_pipeline(model2, cache_dir=huggingface_hub_cache)


@app.get("/")
async def read_root():
    return {"Hello": " er World"}


@app.post("/generate_image/", response_model=None)
async def generate_img(prompt: str, pipe=pipe_initial):
    # print("generating image with pipeline:", pipe)
    if not pipe:
        # in dev we will load the pipe here to avoid slowing app's hot reloading.
        pipe = load_pipeline(model2, cache_dir=huggingface_hub_cache)
    image = img_from_prompt(prompt, pipe, height=256, width=256, guidance_scale=7.5, num_inference_steps=10)
    memory_stream = io.BytesIO()
    image.save(memory_stream, format='PNG')  # write the image to memory instead of the disk, and return it from there
    memory_stream.seek(0)
    return StreamingResponse(memory_stream, media_type="image/png")
