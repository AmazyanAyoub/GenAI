from typing import Union, Optional
from pydantic import BaseModel, Field

class Item(BaseModel):
    id: int
    name: str = Field(default="new Item")
    price: Optional[float] = None
    is_offer: Union[bool, None] = None # is_offer: bool | None = None  # Same as Union[bool, None]