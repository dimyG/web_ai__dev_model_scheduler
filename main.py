import io
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from ml import img_from_prompt, load_pipeline

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load the pipeline in memory on host start, and then reuse it for all images you want to generate.
pipe = load_pipeline()


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.post("/generate_image/")
async def generate_img(prompt: str):
    image = img_from_prompt(prompt, pipe, height=256, width=256, guidance_scale=7.5, num_inference_steps=10)
    memory_stream = io.BytesIO()
    image.save(memory_stream, format='PNG')  # write the image to memory instead of the disk, and return it from there
    memory_stream.seek(0)
    return StreamingResponse(memory_stream, media_type="image/png")
