import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
from typing import Any

import numpy as np
import pandas as pd
from titanic_model import __version__ as model_version
from titanic_model.predict import make_prediction

from fastapi import APIRouter, FastAPI, Request, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.encoders import jsonable_encoder
from fastapi.templating import Jinja2Templates

from app import __version__, schemas
from app.config import settings

app = FastAPI(title=settings.PROJECT_NAME, openapi_url=f"{settings.API_V1_STR}/openapi.json")
# Setup Jinja2 templates
# Define the path to the templates folder
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

api_router = APIRouter()

@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """
    Root Get
    """
    health = schemas.Health(
        name=settings.PROJECT_NAME, api_version=__version__, model_version=model_version
    )

    return health.dict()

@api_router.post("/predict", response_model=schemas.PredictionResults, status_code=200)
async def predict(input_data: schemas.MultipleDataInputs) -> Any:
    """
    Survival predictions with the titanic_model
    """

    input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))
    
    results = make_prediction(input_data=input_df.replace({np.nan: None}))

    if results["errors"] is not None:
        raise HTTPException(status_code=400, detail=json.loads(results["errors"]))

    return results

app.include_router(api_router)

# Root HTML page rendered with Jinja2
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Form-based prediction
@app.post("/predict-form", response_class=HTMLResponse)
async def predict_form(request: Request,
                       pclass: int = Form(...),
                       name: str = Form(...),
                       sex: str = Form(...),
                       age: float = Form(...),
                       sibsp: int = Form(...),
                       parch: int = Form(...),
                       ticket: str = Form(...),
                       fare: float = Form(...),
                       cabin: str = Form(...),
                       embarked: str = Form(...)):

    # Prepare the input data for the prediction based on the DataInputSchema
    input_data = schemas.MultipleDataInputs(
        inputs=[{
            "PassengerId": 1,  # Since we're not using PassengerId in the form, we can hardcode it
            "Pclass": pclass,
            "Name": name,
            "Sex": sex,
            "Age": age,
            "SibSp": sibsp,
            "Parch": parch,
            "Ticket": ticket,
            "Fare": fare,
            "Cabin": cabin if cabin else None,
            "Embarked": embarked
        }]
    )

    # Send data to the predict API using the /predict endpoint
    results = await predict(input_data=input_data)

    if results["errors"]:
        result = f"Error: {results['errors']}"
    else:
        # Get the prediction result
        prediction = results["predictions"]
        result = "Survived" if prediction == 1 else "Did not survive"

    # Render the template with the prediction result
    return templates.TemplateResponse("index.html", {"request": request, "result": result})


# Set all CORS enabled origins
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
