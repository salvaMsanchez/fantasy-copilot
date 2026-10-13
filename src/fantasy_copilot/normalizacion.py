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
MAXIMO_LLAMADAS_A_HERRAMIENTAS = 3
PARECIDO_MINIMO_NOMBRE = 90
PARECIDO_MINIMO_EQUIPO = 60


def _herramienta(nombre: str, descripcion: str, argumento: str, descripcion_argumento: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": nombre,
            "description": descripcion,
            "parameters": {
                "type": "object",
                "properties": {argumento: {"type": "string", "description": descripcion_argumento}},
                "required": [argumento],
            },
        },
    }


HERRAMIENTAS = [
    _herramienta(
        "buscar_en_lista_oficial",
        "Busca si un nombre existe exactamente o casi exactamente en la lista oficial de "
        "jugadores de LaLiga.",
        "nombre",
        "Nombre a buscar en la lista oficial",
    ),
    _herramienta(
        "buscar_por_equipo",
        "Devuelve los jugadores de la lista oficial de un equipo. Úsala cuando la cita mencione "
        "un equipo de LaLiga.",
        "equipo",
        "Nombre del equipo mencionado en la cita",
    ),
]

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

HERRAMIENTAS DISPONIBLES:
1. buscar_en_lista_oficial: úsala cuando no estés seguro de si el nombre existe tal cual.
2. buscar_por_equipo: úsala cuando la cita mencione un equipo de LaLiga.
Sus resultados también llevan el identificador de cada jugador, que puedes devolver aunque no \
esté entre los candidatos.

PROCESO:
1. Analiza el nombre transcrito completo: ¿tiene forma de nombre o apellido de persona?
2. Analiza los candidatos: ¿alguno contiene fonética o textualmente el nombre transcrito?
3. Analiza la cita: ¿confirma o contradice algún candidato? Si menciona un equipo, compáralo \
con el de cada candidato.
4. Si tienes dudas, usa las herramientas: como máximo 3 llamadas en total.

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
    mensajes = [
        {"role": "system", "content": INSTRUCCIONES},
        {"role": "user", "content": json.dumps(peticion, ensure_ascii=False, indent=2)},
    ]
    opciones = {
        "model": modelo,
        "response_format": ESQUEMA_JUGADOR,
        "max_completion_tokens": MAXIMO_TOKENS_RESPUESTA,
        "temperature": TEMPERATURA,
    }

    ejecutadas = 0
    while ejecutadas < MAXIMO_LLAMADAS_A_HERRAMIENTAS:
        respuesta = cliente.chat.completions.create(
            messages=mensajes, tools=HERRAMIENTAS, **opciones
        )
        mensaje = respuesta.choices[0].message
        if not mensaje.tool_calls:
            return json.loads(mensaje.content)["jugador_id"]
        mensajes.append(
            {
                "role": "assistant",
                "content": mensaje.content,
                "tool_calls": [
                    {
                        "id": llamada.id,
                        "type": "function",
                        "function": {
                            "name": llamada.function.name,
                            "arguments": llamada.function.arguments,
                        },
                    }
                    for llamada in mensaje.tool_calls
                ],
            }
        )
        # Toda llamada pedida necesita su mensaje de respuesta, también las que ya no se ejecutan.
        for llamada in mensaje.tool_calls:
            if ejecutadas < MAXIMO_LLAMADAS_A_HERRAMIENTAS:
                ejecutadas += 1
                resultado = _ejecutar_herramienta(
                    llamada.function.name, json.loads(llamada.function.arguments), jugadores
                )
            else:
                resultado = "No ejecutada: se ha alcanzado el máximo de llamadas a herramientas."
            mensajes.append({"role": "tool", "tool_call_id": llamada.id, "content": resultado})

    mensajes.append(
        {
            "role": "user",
            "content": "Ya no puedes usar herramientas. Con lo que has averiguado, devuelve "
            "la respuesta final.",
        }
    )
    respuesta = cliente.chat.completions.create(messages=mensajes, **opciones)
    return json.loads(respuesta.choices[0].message.content)["jugador_id"]


def _ejecutar_herramienta(nombre: str, argumentos: dict, jugadores: list[dict]) -> str:
    if nombre == "buscar_en_lista_oficial":
        return _buscar_en_lista_oficial(argumentos["nombre"], jugadores)
    if nombre == "buscar_por_equipo":
        return _buscar_por_equipo(argumentos["equipo"], jugadores)
    return "Herramienta no reconocida."


def _buscar_en_lista_oficial(nombre: str, jugadores: list[dict]) -> str:
    encontrados = [jugador for jugador in jugadores if jugador["nombre"] == nombre]
    if not encontrados:
        parecido = process.extractOne(
            nombre,
            [jugador["nombre"] for jugador in jugadores],
            scorer=fuzz.WRatio,
            score_cutoff=PARECIDO_MINIMO_NOMBRE,
        )
        if parecido is None:
            return "No hay ninguna coincidencia en la lista oficial."
        encontrados = [jugadores[parecido[2]]]
    return json.dumps(
        [
            {"id": jugador["id"], "nombre": jugador["nombre"], "equipo": jugador["equipo"]}
            for jugador in encontrados
        ],
        ensure_ascii=False,
    )


def _buscar_por_equipo(equipo: str, jugadores: list[dict]) -> str:
    equipos = sorted({nombre for jugador in jugadores for nombre in jugador["equipos"]})
    parecido = process.extractOne(
        equipo, equipos, scorer=fuzz.WRatio, score_cutoff=PARECIDO_MINIMO_EQUIPO
    )
    if parecido is None:
        return "El equipo no se ha encontrado en la lista oficial."
    return json.dumps(
        {
            "equipo": parecido[0],
            "jugadores": [
                {"id": jugador["id"], "nombre": jugador["nombre"]}
                for jugador in jugadores
                if parecido[0] in jugador["equipos"]
            ],
        },
        ensure_ascii=False,
    )
