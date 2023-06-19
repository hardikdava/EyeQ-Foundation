from fastapi import FastAPI
from eyeq_foundation import SAM

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}
