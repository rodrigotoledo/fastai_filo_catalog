from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Photo Finder API - backend base"}
