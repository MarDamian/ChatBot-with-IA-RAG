from mistralai.client import MistralClient
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from mistralai.models.chat_completion import ChatMessage
from fastapi.staticfiles import StaticFiles
import numpy as np
from Backend.helpers import get_text_embedding, initialize_faiss_index
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('API_KEY')

client = MistralClient(api_key=api_key)
pdf_path = os.path.join('Backend/Documento_para_ChatBot.pdf')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#app.mount("/", StaticFiles(directory="Frontend", html=True), name="frontend")

index, chunks = initialize_faiss_index(pdf_path)

class Message(BaseModel):
    message: str

@app.post("/chat")
async def chat(message: Message):
    user_message = message.message
    question_embeddings = np.array([get_text_embedding(user_message)])
    D, I = index.search(question_embeddings, k=2)
    retrieved_chunk = [chunks[i] for i in I.tolist()[0]]

    prompt = f"""
    Contexto relevante:
    ---------------------
    {retrieved_chunk}
    ---------------------
    Instrucciones para la respuesta:
    1. **Brevedad**: Responde de manera concisa y directa. Evita extenderte más de lo necesario.
    2. **Relevancia**: La respuesta debe estar basada en el contexto proporcionado. Si la pregunta no está relacionada con la información del contexto, indica claramente que no hay información relevante disponible.
    3. **Manejo de preguntas irrelevantes**:
       - Si la pregunta no tiene relación con el contexto o no se puede responder con la información proporcionada, informa al usuario de manera cortés y clara que no se puede proporcionar una respuesta útil en base a la información disponible.
       - Sugiere al usuario que reformule su pregunta para que se ajuste mejor al contexto o consulte otras fuentes para obtener una respuesta.
    4. **Actitud y tono**: Mantén una actitud de disponibilidad y amabilidad en todo momento. Evita dar respuestas que puedan parecer despectivas o poco útiles.

    Consulta: {user_message}
    Respuesta:
    """



    def run_mistral(user_message, model="mistral-medium-latest"):
        messages = [ChatMessage(role="user", content=user_message)]
        chat_response = client.chat(model=model, messages=messages)
        return chat_response.choices[0].message.content

    response = run_mistral(prompt)
    return {"response": response}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
