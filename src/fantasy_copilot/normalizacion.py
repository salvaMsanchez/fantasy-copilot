import json
from pathlib import Path

from openai import OpenAIError
from rapidfuzz import fuzz, process

MEDIDAS = (
    fuzz.WRatio,
    fuzz.token_sort_ratio,
    fuzz.token_set_ratio,
    fuzz.partial_ratio,
    fuzz.ratio,
)
CANDIDATOS_POR_MEDIDA = 5
PARECIDO_MINIMO = 30
MAXIMO_TOKENS_RESPUESTA = 500
TEMPERATURA = 0

ESQUEMA_JUGADOR = {
    "type": "json_schema",
    "json_schema": {
        "name": "jugador_resuelto",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"jugador_id": {"type": ["string", "null"]}},
            "required": ["jugador_id"],
            "additionalProperties": False,
        },
    },
}

INSTRUCCIONES = """Eres un experto lingüista y aficionado al fútbol español.

Una transcripción automática de YouTube ha generado un nombre de jugador que puede estar:
- Deformado fonéticamente (por ejemplo, "mastantono" por "Franco Mastantuono").
- En español coloquial (por ejemplo, "Rafiña" por "Raphinha").
- Reducido al apellido o a una parte (por ejemplo, "feng gruver" por "David Affengruber").
- Cortado o fragmentado por la transcripción.

Recibes el nombre transcrito, la cita en la que aparece y una lista de candidatos de la lista \
oficial de LaLiga, cada uno con su identificador, su nombre y su equipo. Devuelve en `jugador_id` \
el identificador del candidato al que se refiere el nombre, o null si ninguno encaja.

PROCESO:
1. Analiza el nombre transcrito completo: ¿tiene forma de nombre o apellido de persona?
2. Analiza los candidatos: ¿alguno contiene fonética o textualmente el nombre transcrito?
3. Analiza la cita: ¿confirma o contradice algún candidato? Si menciona un equipo, compáralo \
con el de cada candidato.

Devuelve null si:
- El nombre transcrito completo no tiene forma de nombre o apellido de persona, aunque alguna \
palabra suelta coincida con un jugador.
- No hay conexión clara entre el nombre transcrito completo y algún candidato: una coincidencia \
parcial de una sola palabra no basta.
- La cita no es coherente con una recomendación sobre un futbolista real.
- Tienes cualquier duda razonable.

Es mejor eliminar un nombre que resolverlo mal."""


def normalizar_insights(
    insights: list[dict],
    jugadores: list[dict],
    *,
    jornada: int,
    temporada: int,
    datos_dir: Path,
    modelo: str,
    cliente,
) -> dict:
    por_id = {jugador["id"]: jugador for jugador in jugadores}
    citas = {}
    for insight in insights:
        citas.setdefault(insight["jugador"], insight.get("quote_original") or "")

    resueltos = {}
    eliminados = []
    for nombre, cita in citas.items():
        try:
            jugador_id = _consultar(nombre, cita, jugadores, modelo=modelo, cliente=cliente)
        except (OpenAIError, ValueError, TypeError, KeyError):
            eliminados.append({"nombre": nombre, "quote": cita, "motivo": "error_api"})
            continue
        if jugador_id in por_id:
            resueltos[nombre] = por_id[jugador_id]
        else:
            eliminados.append({"nombre": nombre, "quote": cita, "motivo": "sin_coincidencia"})

    normalizados = [
        {
            **insight,
            "jugador": resueltos[insight["jugador"]]["nombre"],
            "jugador_id": resueltos[insight["jugador"]]["id"],
            "equipo": resueltos[insight["jugador"]]["equipo"],
            "posicion": resueltos[insight["jugador"]]["posicion"],
            "nombre_transcrito": insight["jugador"],
        }
        for insight in insights
        if insight["jugador"] in resueltos
    ]

    directorio = Path(datos_dir) / str(temporada) / f"j{jornada:02d}"
    directorio.mkdir(parents=True, exist_ok=True)
    _guardar(directorio / "insights_normalizados.json", normalizados)
    _guardar(directorio / "nombres_eliminados.json", eliminados)
    return {"insights": normalizados, "eliminados": eliminados}


def _guardar(fichero: Path, contenido: list[dict]) -> None:
    fichero.write_text(json.dumps(contenido, ensure_ascii=False, indent=2), encoding="utf-8")


def _candidatos(nombre: str, jugadores: list[dict]) -> list[dict]:
    nombres = [jugador["nombre"] for jugador in jugadores]
    candidatos = {}
    for medida in MEDIDAS:
        parecidos = process.extract(
            nombre,
            nombres,
            scorer=medida,
            limit=CANDIDATOS_POR_MEDIDA,
            score_cutoff=PARECIDO_MINIMO,
        )
        for _, _, indice in parecidos:
            jugador = jugadores[indice]
            candidatos[jugador["id"]] = {
                "id": jugador["id"],
                "nombre": jugador["nombre"],
                "equipo": jugador["equipo"],
            }
    return list(candidatos.values())


def _consultar(nombre: str, cita: str, jugadores: list[dict], *, modelo: str, cliente):
    peticion = {
        "nombre_transcrito": nombre,
        "cita": cita or "No disponible",
        "candidatos": _candidatos(nombre, jugadores),
    }
    respuesta = cliente.chat.completions.create(
        model=modelo,
        messages=[
            {"role": "system", "content": INSTRUCCIONES},
            {"role": "user", "content": json.dumps(peticion, ensure_ascii=False, indent=2)},
        ],
        response_format=ESQUEMA_JUGADOR,
        max_completion_tokens=MAXIMO_TOKENS_RESPUESTA,
        temperature=TEMPERATURA,
    )
    return json.loads(respuesta.choices[0].message.content)["jugador_id"]
