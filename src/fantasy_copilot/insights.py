import json
import time
from pathlib import Path

from openai import OpenAIError

from fantasy_copilot.textos import CAMPOS_VIDEO

ACCIONES = ("alinear", "evitar")
INTENTOS = 3
ESPERA_ENTRE_INTENTOS = 1
MAXIMO_TOKENS_RESPUESTA = 6000
TEMPERATURA = 0.1

_TEXTO_O_NULO = {"type": ["string", "null"]}
_CRITERIOS = ("forma_actual", "dinamica_equipo", "calendario_rival", "prediccion_puntos")

ESQUEMA_INSIGHTS = {
    "type": "json_schema",
    "json_schema": {
        "name": "insights_extraidos",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "insights": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "jugador": {"type": "string"},
                            "accion": {"type": "string", "enum": list(ACCIONES)},
                            "confianza": {"type": "number"},
                            "razon_corta": {"type": "string"},
                            "criterios": {
                                "type": "object",
                                "properties": {criterio: _TEXTO_O_NULO for criterio in _CRITERIOS},
                                "required": list(_CRITERIOS),
                                "additionalProperties": False,
                            },
                            "quote_original": _TEXTO_O_NULO,
                        },
                        "required": [
                            "jugador",
                            "accion",
                            "confianza",
                            "razon_corta",
                            "criterios",
                            "quote_original",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["insights"],
            "additionalProperties": False,
        },
    },
}

INSTRUCCIONES = """Eres un experto analista de fantasy football español.
Se te proporciona la transcripción de un vídeo de YouTube de un canal de análisis de LaLiga Fantasy.

INSTRUCCIONES:
1. Extrae todos los insights sobre jugadores concretos que contengan una recomendación clara.
2. Escribe el nombre del jugador EXACTAMENTE como aparece en la transcripción: no lo corrijas ni \
lo normalices.
3. Incluye solo jugadores con una recomendación explícita en el texto: alinear o evitar.
4. La confianza (de 0.0 a 1.0) refleja lo seguro que suena el analista: si duda, bájala; si es \
tajante, súbela.
5. Si un criterio no se menciona, usa null.
6. Si el texto no contiene ninguna recomendación, devuelve una lista vacía."""

_CONTEXTO = """

CONTEXTO DEL VÍDEO:
- Canal: {canal}
- Título: {titulo}
- Fecha: {fecha}
- Jornada: {jornada}"""


class _TextoFallido(Exception):
    def __init__(self, motivo: str) -> None:
        super().__init__(motivo)
        self.motivo = motivo


def extraer_insights(
    textos: list[dict],
    *,
    jornada: int,
    temporada: int,
    datos_dir: Path,
    modelo: str,
    cliente,
    esperar=time.sleep,
) -> dict:
    fichero = Path(datos_dir) / str(temporada) / f"j{jornada:02d}" / "insights.json"
    fichero.parent.mkdir(parents=True, exist_ok=True)
    resultado = _leer_guardado(fichero)
    procesados = {(texto["video_id"], texto["chunk"]) for texto in resultado["procesados"]}

    for texto in textos:
        clave = (texto["video_id"], texto["chunk"])
        if clave in procesados:
            continue
        identidad = {"video_id": texto["video_id"], "chunk": texto["chunk"]}
        resultado["fallidos"] = [
            fallido
            for fallido in resultado["fallidos"]
            if (fallido["video_id"], fallido["chunk"]) != clave
        ]
        try:
            insights = _extraer_de_texto(texto, modelo=modelo, cliente=cliente, esperar=esperar)
        except _TextoFallido as fallo:
            resultado["fallidos"].append({**identidad, "motivo": fallo.motivo})
        else:
            resultado["insights"].extend(insights)
            resultado["procesados"].append(identidad)
            procesados.add(clave)
        _guardar(fichero, resultado)
    return resultado


def _leer_guardado(fichero: Path) -> dict:
    if not fichero.exists():
        return {"insights": [], "procesados": [], "fallidos": []}
    return json.loads(fichero.read_text(encoding="utf-8"))


def _guardar(fichero: Path, resultado: dict) -> None:
    fichero.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")


def _extraer_de_texto(texto: dict, *, modelo: str, cliente, esperar) -> list[dict]:
    insights = _consultar(texto, texto["texto"], modelo=modelo, cliente=cliente, esperar=esperar)
    if insights is None:
        palabras = texto["texto"].split()
        mitad = len(palabras) // 2
        insights = []
        for parte in (palabras[:mitad], palabras[mitad:]):
            de_la_parte = None
            if parte:
                de_la_parte = _consultar(
                    texto, " ".join(parte), modelo=modelo, cliente=cliente, esperar=esperar
                )
            if de_la_parte is None:
                raise _TextoFallido("respuesta_cortada")
            insights.extend(de_la_parte)

    fuente = {**{campo: texto[campo] for campo in CAMPOS_VIDEO}, "chunk": texto["chunk"]}
    return [{**insight, "fuente": dict(fuente)} for insight in insights if _es_valido(insight)]


def _consultar(texto: dict, transcripcion: str, *, modelo: str, cliente, esperar):
    """Devuelve los insights de la respuesta, o None si llegó cortada por límite de tokens."""
    mensajes = [
        {"role": "system", "content": INSTRUCCIONES + _CONTEXTO.format(**texto)},
        {"role": "user", "content": transcripcion},
    ]
    for intento in range(1, INTENTOS + 1):
        try:
            respuesta = cliente.chat.completions.create(
                model=modelo,
                messages=mensajes,
                response_format=ESQUEMA_INSIGHTS,
                max_completion_tokens=MAXIMO_TOKENS_RESPUESTA,
                temperature=TEMPERATURA,
            )
            eleccion = respuesta.choices[0]
            if eleccion.finish_reason == "length":
                return None
            insights = json.loads(eleccion.message.content)["insights"]
            if not isinstance(insights, list):
                raise TypeError("insights no es una lista")
            return insights
        except OpenAIError:
            motivo = "error_api"
        except (ValueError, TypeError, KeyError):
            motivo = "respuesta_no_valida"
        if intento < INTENTOS:
            esperar(ESPERA_ENTRE_INTENTOS)
    raise _TextoFallido(motivo)


def _es_valido(insight) -> bool:
    if not isinstance(insight, dict) or insight.get("accion") not in ACCIONES:
        return False
    confianza = insight.get("confianza")
    return (
        isinstance(confianza, int | float)
        and not isinstance(confianza, bool)
        and 0 <= confianza <= 1
    )
