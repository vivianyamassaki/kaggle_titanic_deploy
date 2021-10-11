import os
import logging

import pandas as pd
from fastapi.responses import JSONResponse
import uvicorn
from fastapi import FastAPI
from typing import List

from predictor.pipeline.predictor import Predictor
from predictor.serializers.response import ResponseSerializer
from predictor.serializers.passenger import Passenger
from predictor.settings import CONFIGS
from feature_engineering.feature_engineering_pipeline import FeatureEngineering

logger = logging.getLogger(__name__)

app = FastAPI()


@app.get('/health', status_code=200)
def health_check():
    """application health check"""
    return JSONResponse({"status": "healthy"})


@app.post('/predict')
async def predict(data: List[Passenger]):
    """
    Use input from request to transform data and make predictions
    :param data: list of passenger used as input to model
    :return: predictions serialized as json
    """
    ids, raw_data = to_dataframe(data)
    processed_data = FeatureEngineering(CONFIGS['numerical_features']).transform(raw_data)
    predictions = Predictor(CONFIGS['model_path']).predict(processed_data)
    return ResponseSerializer().serialize(ids, predictions)


def to_dataframe(data: List[Passenger]) -> pd.DataFrame:
    logger.info('transforming json request to dataframe...')
    id_column = CONFIGS['id']
    df = pd.DataFrame([vars(passenger) for passenger in data])
    logger.info(f'request entities ids: {df[id_column].values}')
    return df[id_column], df.drop(id_column, axis=1)


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host='0.0.0.0',
        port=os.environ.get('PORT', 8000),
        log_level=os.environ.get('LOGLEVEL', 'info').lower()
    )
