from pydantic import BaseModel
from typing import Optional


class Passenger(BaseModel):
    PassengerId: int
    Pclass: int
    Sex: str
    Age: int
    Fare: Optional[float]
