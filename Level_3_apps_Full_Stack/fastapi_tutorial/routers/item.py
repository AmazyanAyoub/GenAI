from fastapi import APIRouter, Path
from models.item import Item

router = APIRouter()

items = [
    {
        "id": 1,
        "name": "iphone",
        "price": 950,
        "is_offer": False
    },
    {
        "id": 2,
        "name": "samsung",
        "price": 750,
        "is_offer": True
    }
]

@router.get("/items/{item_id}")
def read_item(item_id: int = Path(..., gt=0), q: str | None = None):
    return {"item_id": item_id, "q": q}

@router.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {
        "id": item_id,
        "name": item.name,
        "price": item.price,
        "is_offer": item.is_offer
    }

@router.put("/items/update/{item_id}")
def second_update(item_id: int, item: Item):
    for index, item in enumerate(items):
        if item.id == item_id:
            items[index]["name"] = item.name
            items[index]["price"] = item.price
            items[index]["is_offer"] = item.is_offer
    return items

@router.get("/items/")
def get_items():
    return items

@router.post("/items/")
def create_item(item: Item):
    items.append(item)
    return items

@router.delete("/items/{item_id}")
def delete_item(item_id: int):
    for item in items:
        if item["id"] == item_id:
            items.remove(item)
    return item