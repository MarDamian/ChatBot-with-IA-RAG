# Chatbot Project

Este proyecto es un chatbot que utiliza la API de MistralAI y Retrieval-Augmented Generation (RAG) para proporcionar respuestas precisas y contextualmente relevantes. El proyecto está dividido en un frontend y un backend, con la siguiente estructura de archivos:


## Requisitos

Para ejecutar este proyecto, necesitarás tener instalado:

- Python 3.12
- Las dependencias listadas en `requirements.txt`

## Instalación

1. Clona el repositorio en tu máquina local.
2. Crea y activa un entorno virtual (opcional pero recomendado).
3. Instala las dependencias:

    ```bash
    pip install -r requirements.txt
    ```

4. Configura las variables de entorno en los archivos `.env` en la raíz y en el directorio `Backend`.

## Uso

### Backend

El backend está implementado en Python y maneja la lógica del chatbot y las interacciones con la API de MistralAI.

- `assistant.py`: Contiene la lógica principal del chatbot.
- `helpers.py`: Funciones auxiliares utilizadas por `assistant.py`.
- `Documento_para_ChatBot.pdf` y `Documento_para_ChatBot.txt`: Documentos de referencia utilizados por el chatbot.

Para iniciar el backend:

```bash
python main.py
