from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel


from pydantic import BaseModel, Field, validator

class Item(BaseModel):
    text: str = Field(..., min_length=1, max_length=500, description="Текст для анализа тональности")
    
    @validator('text')
    def text_not_empty(cls, v):
        if not v or v.strip() == '':
            raise ValueError('Текст не может быть пустым')
        return v.strip()


app = FastAPI()
classifier = pipeline("sentiment-analysis")


@app.get("/")
def root():
    return {"FastApi service started!"}


@app.get("/{text}")
def get_params(text: str):
    return classifier(text)


@app.post("/predict/")
def predict(item: Item):
    return classifier(item.text)
