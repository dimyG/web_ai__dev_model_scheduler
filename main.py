from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.post("/generate_img/")
async def generate_img(prompt: str):
    return {"prompt": prompt}

