from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from fastapi.staticfiles import StaticFiles
import numpy as np

from Backend.helpers import get_text_embedding, initialize_faiss_index
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('API_KEY')

# Usa api_key en tu código
client = MistralClient(api_key=api_key)


app = FastAPI()

# Permitir solicitudes CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ruta al archivo PDF local
pdf_path = os.path.join('Backend/Documento_para_ChatBot.pdf')

class Message(BaseModel):
    message: str
    
# Monta los archivos estáticos del frontend
#app.mount("/", StaticFiles(directory="Frontend", html=True), name="frontend")

# Inicialización de la base de datos vectorial
index, chunks = initialize_faiss_index(pdf_path)


@app.post("/chat")
async def chat(message: Message):
    user_message = message.message

    question_embeddings = np.array([get_text_embedding(user_message)])
    D, I = index.search(question_embeddings, k=2)
    retrieved_chunk = [chunks[i] for i in I.tolist()[0]]
    
    prompt = f"""
    La información del contexto se encuentra a continuación.
    ---------------------
    {retrieved_chunk}
    ---------------------
    Sea lo más breve posible sin extenderse demasiado con la respuesta.
    Si no encuentras la resupuesta solicitada informa que no tienes informacion al respecto.
    Mantente con una actitud de disponibilidad.
    Si la pregunta no tiene sentido para ti, sugiere que reenvies la pregunta ya que no entendiste y no des informacion que no este relacionada.
    
    Consulta: {user_message}
    Respuesta:
    """

    def run_mistral(user_message, model="mistral-medium-latest"):
        messages = [
            ChatMessage(role="user", content=user_message)
        ]
        chat_response = client.chat(
            model=model,
            messages=messages
        )
        return chat_response.choices[0].message.content

    response = run_mistral(prompt)
    return {"response": response}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
    #uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))