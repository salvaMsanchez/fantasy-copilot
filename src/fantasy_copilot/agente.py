import json
import time

from openai import OpenAIError

UMBRAL_CONFIANZA = 0.4
TOP_K = 10
PREGUNTA = "¿Qué dicen los expertos sobre {nombre} en fantasy football?"
TAMANO_ONCE = 11
INTENTOS = 3
ESPERA_ENTRE_INTENTOS = 1
MAXIMO_TOKENS_RESUMEN = 500
MAXIMO_TOKENS_RECOMENDACION = 2000
TEMPERATURA_RESUMEN = 0.1
TEMPERATURA_RECOMENDACION = 0.2


def _esquema(nombre: str, propiedades: dict) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": nombre,
            "strict": True,
            "schema": {
                "type": "object",
                "properties": propiedades,
                "required": list(propiedades),
                "additionalProperties": False,
            },
        },
    }


ESQUEMA_RESUMEN = _esquema(
    "resumen_jugador",
    {
        "jugador": {"type": "string"},
        "posicion": {"type": "string"},
        "accion_recomendada": {"type": "string", "enum": ["alinear", "evitar", "sin_datos"]},
        "confianza_media": {"type": "number"},
        "resumen": {"type": "string"},
        "fuentes": {"type": "array", "items": {"type": "string"}},
    },
)

ESQUEMA_RECOMENDACION = _esquema(
    "recomendacion_once",
    {
        "once_recomendado": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "jugador": {"type": "string"},
                    "posicion": {"type": "string"},
                    "confianza": {"type": "number"},
                    "justificacion": {"type": "string"},
                },
                "required": ["jugador", "posicion", "confianza", "justificacion"],
                "additionalProperties": False,
            },
        },
        "advertencias": {"type": "array", "items": {"type": "string"}},
        "resumen": {"type": "string"},
    },
)

INSTRUCCIONES_RESUMEN = """Eres un experto analista de fantasy football español para LaLiga.

Recibes un jugador de la plantilla del usuario, su posición de fantasy, la jornada y los insights \
de analistas de YouTube sobre él. Basándote solo en esos insights, resume en dos o tres líneas lo \
más relevante para esta jornada.

- accion_recomendada: "alinear" o "evitar", según lo que recomienden los insights en conjunto; \
"sin_datos" si no permiten decidir.
- confianza_media: la media de la confianza de los insights, entre 0 y 1.
- fuentes: cada fuente como "canal - Jornada N", sin repetir.
- jugador y posicion: los mismos que recibes."""

INSTRUCCIONES_RECOMENDACION = """Eres un experto analista de fantasy football español para LaLiga.

Recibes la jornada y el resumen de cada jugador de la plantilla del usuario. Recomienda el once \
óptimo para esta jornada:
- Exactamente 11 jugadores distintos, todos de la plantilla, con el nombre tal como aparece en \
los resúmenes.
- Equilibrio por posiciones: 1 portero, 4 defensas, 3 o 4 centrocampistas y 2 o 3 delanteros.
- Prioriza a los jugadores con mayor confianza y justifica los casos dudosos.
- En advertencias, los jugadores con dudas o sin datos.
- En resumen, una visión general del once en dos o tres líneas."""


class ErrorRecomendacion(Exception):
    pass


def recomendar_once(
    plantilla: list[dict],
    *,
    jornada: int,
    temporada: int,
    indice,
    modelo: str,
    cliente,
    esperar=time.sleep,
) -> dict:
    resumenes = []
    avisos = []
    for jugador in plantilla:
        insights = [
            insight
            for insight in indice.consultar(
                PREGUNTA.format(nombre=jugador["nombre"]),
                jornada=jornada,
                temporada=temporada,
                top_k=TOP_K,
            )
            if insight["metadatos"]["confianza"] >= UMBRAL_CONFIANZA
        ]
        if not insights:
            resumenes.append(_sin_datos(jugador))
            continue
        try:
            resumenes.append(
                _pedir(
                    INSTRUCCIONES_RESUMEN,
                    _peticion_resumen(jugador, jornada, insights),
                    ESQUEMA_RESUMEN,
                    MAXIMO_TOKENS_RESUMEN,
                    TEMPERATURA_RESUMEN,
                    modelo=modelo,
                    cliente=cliente,
                    esperar=esperar,
                )
            )
        except _PeticionFallida as error:
            resumenes.append(_sin_datos(jugador))
            avisos.append(f"No se pudo resumir a {jugador['nombre']}: {error}")

    peticion = json.dumps(
        {"jornada": jornada, "resumenes": resumenes}, ensure_ascii=False, indent=2
    )
    try:
        recomendacion = _pedir(
            INSTRUCCIONES_RECOMENDACION,
            peticion,
            ESQUEMA_RECOMENDACION,
            MAXIMO_TOKENS_RECOMENDACION,
            TEMPERATURA_RECOMENDACION,
            modelo=modelo,
            cliente=cliente,
            esperar=esperar,
        )
    except _PeticionFallida as error:
        raise ErrorRecomendacion(
            f"No se pudo obtener la recomendación del once: {error}"
        ) from error

    avisos.extend(_avisos_del_once(recomendacion["once_recomendado"], plantilla))
    recomendacion["advertencias"] = [*recomendacion["advertencias"], *avisos]
    return {"resumenes": resumenes, "recomendacion": recomendacion}


class _PeticionFallida(Exception):
    pass


def _sin_datos(jugador: dict) -> dict:
    return {
        "jugador": jugador["nombre"],
        "posicion": jugador["posicion"],
        "accion_recomendada": "sin_datos",
        "confianza_media": 0,
        "resumen": "",
        "fuentes": [],
    }


def _peticion_resumen(jugador: dict, jornada: int, insights: list[dict]) -> str:
    return json.dumps(
        {
            "jugador": jugador["nombre"],
            "posicion": jugador["posicion"],
            "jornada": jornada,
            "insights": [
                {
                    "accion": insight["metadatos"]["accion"],
                    "confianza": insight["metadatos"]["confianza"],
                    "canal": insight["metadatos"]["canal"],
                    "jornada": insight["metadatos"]["jornada"],
                    "texto": insight["texto"],
                }
                for insight in insights
            ],
        },
        ensure_ascii=False,
        indent=2,
    )


def _pedir(
    instrucciones: str,
    contenido: str,
    esquema: dict,
    maximo_tokens: int,
    temperatura: float,
    *,
    modelo: str,
    cliente,
    esperar,
) -> dict:
    for intento in range(1, INTENTOS + 1):
        try:
            respuesta = cliente.chat.completions.create(
                model=modelo,
                messages=[
                    {"role": "system", "content": instrucciones},
                    {"role": "user", "content": contenido},
                ],
                response_format=esquema,
                max_completion_tokens=maximo_tokens,
                temperature=temperatura,
            )
            return json.loads(respuesta.choices[0].message.content)
        except OpenAIError:
            motivo = "error_api"
        except (ValueError, TypeError, KeyError, IndexError):
            motivo = "respuesta_no_valida"
        if intento < INTENTOS:
            esperar(ESPERA_ENTRE_INTENTOS)
    raise _PeticionFallida(motivo)


def _avisos_del_once(once: list[dict], plantilla: list[dict]) -> list[str]:
    avisos = []
    nombres = [jugador["jugador"] for jugador in once]
    distintos = set(nombres)
    if len(nombres) != TAMANO_ONCE or len(distintos) != TAMANO_ONCE:
        avisos.append(
            f"El once recomendado tiene {len(nombres)} jugadores ({len(distintos)} distintos) "
            f"y deben ser {TAMANO_ONCE} distintos"
        )
    de_la_plantilla = {jugador["nombre"] for jugador in plantilla}
    for nombre in dict.fromkeys(nombres):
        if nombre not in de_la_plantilla:
            avisos.append(f"{nombre} no está en la plantilla")
    return avisos
