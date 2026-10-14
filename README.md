# Fantasy Copilot — Agente analista con IA

Recomienda alineaciones de LaLiga Fantasy a partir de lo que dicen los analistas en YouTube.

## ¿Qué hace?

Para una jornada, busca los vídeos de los canales de análisis fantasy, descarga sus transcripciones y extrae con OpenAI recomendaciones estructuradas por jugador (alinear o evitar, con confianza y motivo). Los nombres que deforman los subtítulos automáticos ("Rafiña") se resuelven a jugadores oficiales de la lista de Transfermarkt. Con tu plantilla, resume lo que se dice de cada jugador y propone el once.

## Pipeline

```
Transfermarkt ─▶ Lista oficial de jugadores ─────────────┐
                                                         ▼
YouTube ─▶ Vídeos de la jornada ─▶ Transcripciones ─▶ Insights ─▶ Nombres resueltos
                                                                         │
Tu plantilla ─▶ Validada contra la lista oficial ─▶ Agente ◀─ Índice (ChromaDB) ◀┘
                                                     │
                                                     ▼
                                              Once recomendado
```

## Requisitos

- Python 3.11 o posterior.
- Clave de [OpenAI](https://platform.openai.com).
- Clave de [YouTube Data API v3](https://console.cloud.google.com), de Google Cloud.
- Opcional: proxy residencial de [Webshare](https://www.webshare.io), si YouTube bloquea la descarga de transcripciones desde tu IP.

## Instalación

```bash
git clone https://github.com/salvaMsanchez/fantasy-copilot.git
cd fantasy-copilot
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

Después, rellena las claves en `.env`. El fichero `.env` no se sube al repositorio.

## Variables de entorno

Se leen del entorno o del fichero `.env` del directorio desde el que se ejecuta el comando.

| Variable | Obligatoria | Uso | Por defecto |
|----------|-------------|-----|-------------|
| `OPENAI_API_KEY` | Sí | Extracción, resolución de nombres, embeddings y recomendación | — |
| `YOUTUBE_API_KEY` | Sí, para `ingesta` | Búsqueda de vídeos | — |
| `WEBSHARE_PROXY_USERNAME` | No | Proxy para las transcripciones; se definen las dos o ninguna | — |
| `WEBSHARE_PROXY_PASSWORD` | No | Proxy para las transcripciones; se definen las dos o ninguna | — |
| `FANTASY_DATOS_DIR` | No | Carpeta donde se guardan los datos | `datos` |
| `FANTASY_MODELO_EXTRACCION` | No | Modelo que extrae los insights | `gpt-5.4-nano` |
| `FANTASY_MODELO_NORMALIZACION` | No | Modelo que resuelve los nombres | `gpt-5.4-nano` |
| `FANTASY_MODELO_AGENTE` | No | Modelo que resume y recomienda el once | `gpt-5.4-mini` |
| `FANTASY_MODELO_EMBEDDINGS` | No | Modelo de embeddings del índice | `text-embedding-3-small` |

## Uso

La temporada es el año en que empieza: la 2026-27 es `2026`. La jornada va de 1 a 38.

### 1. Ingesta de una jornada

```bash
fantasy-copilot ingesta --jornada 12 --temporada 2026
```

Descarga la lista oficial de jugadores de la temporada (solo la primera vez), busca los vídeos de la jornada, descarga sus transcripciones, extrae los insights y resuelve los nombres. Todo queda en `datos/`, con los insights resueltos en `datos/2026/j12/insights_normalizados.json`. Termina con un resumen de cifras: vídeos, transcripciones obtenidas y no disponibles, insights extraídos, textos fallidos, insights normalizados y nombres eliminados.

Si se vuelve a ejecutar para la misma jornada, cada etapa aprovecha lo que ya guardó: solo se procesa lo nuevo y lo que falló.

### 2. Recomendación del once

```bash
cp plantilla.ejemplo.json plantilla.json
fantasy-copilot recomendar --jornada 12 --temporada 2026 --plantilla plantilla.json
```

Necesita haber ejecutado antes la ingesta de esa jornada y temporada. Imprime los once jugadores con su posición, confianza y justificación, seguidos de las advertencias y de un resumen.

### Formato de la plantilla

Un fichero JSON con la lista de jugadores de tu equipo, como `plantilla.ejemplo.json`:

```json
[
  {"nombre": "Augusto Batalla", "posicion": "portero"},
  {"nombre": "Andrei Ratiu", "posicion": "defensa"},
  {"nombre": "Aleix Febas", "posicion": "centrocampista"},
  {"nombre": "Vedat Muriqi", "posicion": "delantero"}
]
```

- Al menos 11 jugadores, sin repetir ninguno.
- `posicion` es la de fantasy: `portero`, `defensa`, `centrocampista` o `delantero`.
- `nombre` es el de la lista oficial de Transfermarkt. Se compara sin distinguir mayúsculas ni tildes: `"goncalo guedes"` vale por `Gonçalo Guedes`. Si un nombre no está, coincide con dos jugadores o se repite, el comando termina con un mensaje que lo indica.

## Fuentes de datos y uso

- **Transcripciones:** proceden de vídeos públicos de YouTube de los canales de análisis [Carrasco](https://www.youtube.com/channel/UCu7ZOzCUgWpvvJzYAdnw0_Q) y [Jornada Perfecta](https://www.youtube.com/channel/UCvMw41zGXyYYPDvyXcHZfug). Cada insight guarda el enlace al vídeo del que sale.
- **Jugadores:** la lista oficial de jugadores de LaLiga procede de [Transfermarkt](https://www.transfermarkt.es), una vez por temporada y con pausas entre peticiones.
- **Uso:** educativo y no comercial, dentro de un programa de formación. No se cobra por nada.
- **Sin redistribución:** el repositorio no incluye transcripciones ni datos descargados; cada uno los obtiene al ejecutar la ingesta, y quedan en `datos/`, excluida de git.
- **Autoría:** los análisis y opiniones son de sus creadores. Este proyecto solo los resume y enlaza a los vídeos originales.

## Notebook original

`agente_analista.ipynb` es el prototipo con el que empezó el proyecto, pensado para Google Colab, y la base del paquete actual. Se conserva como referencia: incluye un chat con Gradio que todavía no está en el paquete. Para ejecutar el proyecto se usan los comandos de arriba.

## Desarrollo

```bash
pip install -e ".[dev]"
python -m pytest
ruff check src tests
```

## Tecnologías

- Python · argparse
- OpenAI: chat completions con salida estructurada y herramientas, embeddings
- ChromaDB, en memoria
- YouTube Data API v3 (`google-api-python-client`) · `youtube-transcript-api` · Webshare
- Transfermarkt (`requests` y Beautiful Soup) · RapidFuzz

## Autor

Salvador Moreno Sánchez  
Programa Técnico Avanzado en Ingeniería de IA
