# Fantasy Copilot — Agente analista con IA

Sistema de recomendación de alineaciones para LaLiga Fantasy basado en extracción automática de contenido de YouTube, RAG y modelos de lenguaje.

## ¿Qué hace?

Extrae transcripciones de los principales canales de análisis fantasy en YouTube, procesa el contenido con OpenAI para obtener insights estructurados por jugador, los indexa en una base de datos vectorial y genera recomendaciones de alineación personalizadas a través de un chatbot conversacional.

## Pipeline

YouTube → Transcripciones → Insights JSON → RAG (ChromaDB) → Agente Analista → Chatbot

## Requisitos

- Python 3.10+
- Cuenta en [OpenAI](https://platform.openai.com) — modelos `gpt-4o-mini` y `text-embedding-3-small`
- Cuenta en [Google Cloud](https://console.cloud.google.com) — YouTube Data API v3
- Cuenta en [Webshare](https://www.webshare.io) — proxy residencial para descarga de transcripciones

## Instalación

```bash
pip install youtube-transcript-api==1.2.4
pip install openai
pip install chromadb
pip install google-api-python-client
pip install gradio
pip install tiktoken
```

## Ejecución

El proyecto está desarrollado en Google Colab. Ejecuta los bloques en orden:

| Bloque | Descripción |
|--------|-------------|
| 1 | Instalación de librerías |
| 2 | Imports y API keys |
| 3 | Obtención de videos de YouTube |
| 4 | Extracción de transcripciones |
| 5 | Limpieza y chunking |
| 6 | Extracción de insights con OpenAI |
| 7 | Indexación en ChromaDB |
| 8 | Agente analista — recomendación de plantilla |
| 9 | Chatbot con Gradio |

Si ya tienes `insights_j38.json` generado, los bloques 4 y 6 lo cargan automáticamente sin realizar nuevas llamadas a las APIs.

## Archivos necesarios entre sesiones

- `transcripciones_j38.json` — transcripciones en crudo de los videos
- `insights_j38.json` — insights estructurados extraídos por OpenAI

## Tecnologías

- YouTube Data API · youtube-transcript-api · Webshare
- OpenAI: GPT-4o-mini · text-embedding-3-small
- ChromaDB · Gradio

## Mejoras futuras

- Base de datos oficial de jugadores de LaLiga + fuzzy matching para normalización de nombres
- Fine-tuning propio para NER de jugadores y extracción de JSON estructurado
- Pipeline en tiempo real ejecutado automáticamente cada jornada
- Sistema multiagente con orquestador + agentes de estadísticas y noticias
- Integración con apps de fantasy (Biwenger, LaLiga Fantasy)
- Migración a LangChain con memoria conversacional y agentes ReAct

## Autor

Salvador Moreno Sánchez  
Programa Técnico Avanzado en Ingeniería de IA
