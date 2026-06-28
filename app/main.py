"""FastAPI Main Entrypoint."""

from fastapi import FastAPI

app = FastAPI(title="ShieldFlow Risk Evaluation Gateway")

@app.get("/health")
def health_check():
    return {"status": "healthy"}
