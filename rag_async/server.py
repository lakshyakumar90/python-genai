from fastapi import FastAPI, Query
from .client.rq_client import queue
from .queues.worker import process_query

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/chat")
async def chat(query: str = Query(..., description="The chat query of the user")):
    job = queue.enqueue(process_query, query)
    return {"status": "success", "job_id": job.id}


@app.get("/result")
def get_result(job_id: str = Query(..., description="The job id of the query")):
    job = queue.fetch_job(job_id)
    result = job.return_value()
    return {"status": "success", "result": result}
