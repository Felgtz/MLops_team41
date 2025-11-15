from fastapi import FastAPI

app = FastAPI(
    title="MLops Team 41 - Demo API",
    version="0.1.0"
)

@app.get("/")
def read_root():
    return {"message": "Hola, el contenedor ml-service está funcionando 😊"}

@app.get("/health")
def health_check():
    return {"status": "ok"}
