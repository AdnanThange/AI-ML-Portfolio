from pydantic import BaseModel

class BostonInput(BaseModel):
    RM: float
    LSTAT: float
    PTRATIO: float
