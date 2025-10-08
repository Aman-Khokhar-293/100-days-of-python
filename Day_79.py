from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Day 79 – FastAPI Demo")

class Item(BaseModel):
    name: str
    price: float
    in_stock: bool

@app.get("/")
def home():
    return {"message": "Welcome to FastAPI!"}

@app.get("/hello/{name}")
def say_hello(name: str):
    return {"message": f"Hello, {name}!"}

@app.post("/items/")
def create_item(item: Item):
    return {"item_name": item.name, "price": item.price, "in_stock": item.in_stock}

# Run using:
# uvicorn Day_79:app --reload
# Swagger Docs: http://127.0.0.1:8000/docs
# ReDoc: http://127.0.0.1:8000/redoc
