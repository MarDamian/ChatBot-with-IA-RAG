from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import numpy as np
import faiss
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('API_KEY')

client = MistralClient(api_key=api_key)
pdf_path = os.path.join('Backend/Documento_para_ChatBot.pdf')

def read_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ''.join(page.extract_text() + '\n' for page in reader.pages)
        return text
    except Exception as e:
        print(f"Error: No se pudo leer el archivo PDF. {e}")
        return None

def get_text_embedding(input):
    try:
        response = client.embeddings(model="mistral-embed", input=input)
        return response.data[0].embedding
    except Exception as e:
        print(f"Error: No se pudo obtener embedding. {e}")
        return None

def save_text_to_file(text, file_path):
    try:
        with open(file_path, 'w') as f:
            f.write(text)
    except Exception as e:
        print(f"Error: No se pudo guardar el archivo de texto. {e}")

def initialize_faiss_index(pdf_path):
    text = read_pdf(pdf_path)
    if not text:
        return None, None

    text_path = os.path.join('Backend/Documento_para_ChatBot.txt')
    save_text_to_file(text, text_path)

    chunk_size = 2048
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    text_embeddings = np.array([get_text_embedding(chunk) for chunk in chunks if get_text_embedding(chunk) is not None])
    if text_embeddings.size == 0:
        return None, None

    d = text_embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(text_embeddings)

    return index, chunks
