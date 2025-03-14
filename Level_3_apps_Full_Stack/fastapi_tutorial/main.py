from fastapi import FastAPI
from routers.item import router as router_items


app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

app.include_router(router_items)