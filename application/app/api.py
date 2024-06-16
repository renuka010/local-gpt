import os
import json
import requests



from logs import logger
from settings import MARQO_URL, MARQO_INDEX_NAME, MARQO_LIMIT

from llm_loader import pipe

# session
session = requests.session()


def fetch_from_marqo(query):

    headers = {'Content-Type': 'application/json'}
    data = {
        "q": query,
        "limit": MARQO_LIMIT
    }
    response = session.post(f"{MARQO_URL}/indexes/{MARQO_INDEX_NAME}/search", headers=headers, data=json.dumps(data)).json()
    hits = response["hits"]

    result = []
    
    for hit in hits:
        json_metadata = json.loads(hit["metadata"])
        
        temp = {
            "metadata": {
                "page_number": json_metadata["page_label"],
                "file_name": json_metadata["file_name"]
            },
            "context": hit["text"],
            "summary": hit["_highlights"]["text"]
        }

        result.append(temp)

    return result

def query_llm(query, docs):
    ## try1 gpt like prompt
    # json_resp = {
    #     "metadata": {
    #         "page_number": "all page_numbers",
    #         "file_name": "filenames"
    #     },
    #     "summary": "consolidated summary generated"
    # }
    # prompt = f"The provided data conatins metadata and texts, your objective: \
    #     use metadata and text to genetate a response in json containing the following \
    #     {json_resp}. Make sure that if the {query} does not relate to given context \
    #     give an empty json response. The given docs are {docs}"

    ## try 2 manipultion docs
    metadata = []
    summary = []

    for item in docs:
        metadata.append(item["metadata"])
        summary.append(item["context"])

    prompt = f"summarise the following {summary}"

    global pipe
    sequences = pipe(
        prompt,
        do_sample=True,
        max_new_tokens=100, 
        temperature=0.7, 
        top_k=50, 
        top_p=0.95,
        num_return_sequences=1,
    )

    gen = sequences[0]['generated_text']
    logger.info(f"generated_response: {gen}")
    try:
        cleaned = json.loads(gen)
        response = cleaned.split("\n")[-1]
    except:
        response = gen.split("\\n")[-1]

    return metadata, response





