from fastapi.responses import JSONResponse


class ResponseSerializer:

    @staticmethod
    def serialize(ids, predictions):
        return JSONResponse(
            [{'PassengerId': p_id, 'prediction': int(prediction)} for p_id, prediction in zip(ids, predictions)]
        )
