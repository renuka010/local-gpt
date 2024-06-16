# imports


# FastAPI imports
from fastapi import APIRouter, Response, Depends, status,HTTPException

# Package imports
from logs import logger
from schemas import ResponseModel, FetchRequestModel
from api import fetch_from_marqo, query_llm





# Initialize logger instance


# Init Env Objects

# Router Init
router = APIRouter()

logger


@router.get("/")
def index():
    payload = {
        "title": "local-llm-gpt",
        "version": "rel-1.0.1"
    }
    return {"info": payload}



@router.get("/generate", status_code=status.HTTP_200_OK)
async def app(response: Response, data: FetchRequestModel = Depends(FetchRequestModel)):

    query = data.query
    docs = fetch_from_marqo(query=query)
    metadata, resp = query_llm(query, docs)
    # logger.info(f"marqo_data: {docs}")
    try:
        
        logger.info(f"in try block")

        return ResponseModel(
                        status="Ok",
                        code="200",
                        message="Application Service Running",
                        result = {
                            "metadata": metadata,
                            "summary": resp
                        }
                )

    except Exception as e:
        logger.debug(f"Check Inputstring: {e}")
        response.status_code = status.HTTP_400_BAD_REQUEST
        raise HTTPException(status_code=response.status_code, detail="Check searchString")

