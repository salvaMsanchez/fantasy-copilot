import json
from collections import Counter
from types import SimpleNamespace

import pytest

from fantasy_copilot.evaluacion import evaluar_once
from fantasy_copilot.plantilla import ErrorPlantilla

JORNADA = 12
TEMPORADA = 2026


def jugador(jugador_id, posicion):
    return {
        "nombre": f"Jugador {jugador_id}",
        "jugador_id": jugador_id,
        "equipo": "Equipo",
        "posicion": posicion,
    }


def plantilla_de(porteros=(), defensas=(), centrocampistas=(), delanteros=()):
    return [
        *(jugador(i, "portero") for i in porteros),
        *(jugador(i, "defensa") for i in defensas),
        *(jugador(i, "centrocampista") for i in centrocampistas),
        *(jugador(i, "delantero") for i in delanteros),
    ]


PLANTILLA = plantilla_de(
    porteros=["p1", "p2"],
    defensas=["d1", "d2", "d3", "d4", "d5"],
    centrocampistas=["c1", "c2", "c3", "c4"],
    delanteros=["f1", "f2", "f3", "f4"],
)
POSICION = {j["jugador_id"]: j["posicion"] for j in PLANTILLA}
PUNTOS = {
    "p1": 3, "p2": 8,
    "d1": 6, "d2": 1, "d3": 4, "d4": 2, "d5": 7,
    "c1": 9, "c2": 0, "c3": 5, "c4": 3,
    "f1": 2, "f2": 10, "f3": -1, "f4": 4,
}  # fmt: skip
ONCE_RECOMENDADO = ["p1", "d1", "d2", "d3", "d4", "c1", "c2", "c3", "c4", "f1", "f2"]

# Plantilla con un solo portero y tres centrocampistas: la única formación es 1-4-3-3 y la
# elección simple deja fuera a uno de los cuatro delanteros, el de peor media.
PLANTILLA_SIMPLE = plantilla_de(
    porteros=["p1"],
    defensas=["d1", "d2", "d3", "d4"],
    centrocampistas=["c1", "c2", "c3"],
    delanteros=["f1", "f2", "f3", "f4"],
)
ONCE_SIMPLE = ["p1", "d1", "d2", "d3", "d4", "c1", "c2", "c3", "f1", "f2", "f3"]


class IndiceVacio:
    """Hace de índice sin insights: la recomendación no pide resúmenes, solo el once."""

    def consultar(self, pregunta, *, jornada, temporada, top_k=10):
        return []


class OpenAIFalso:
    """Hace de cliente de OpenAI: a toda petición responde con el once fijado."""

    def __init__(self, nombres):
        self.respuesta = {
            "once_recomendado": [
                {
                    "jugador": nombre,
                    "posicion": "delantero",
                    "confianza": 0.8,
                    "justificacion": "En racha",
                }
                for nombre in nombres
            ],
            "advertencias": [],
            "resumen": "Once de prueba.",
        }
        self.peticiones = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._crear))

    def _crear(self, **peticion):
        self.peticiones.append(peticion)
        mensaje = SimpleNamespace(
            role="assistant",
            content=json.dumps(self.respuesta, ensure_ascii=False),
            refusal=None,
            tool_calls=None,
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=mensaje, finish_reason="stop")])


def cliente_con(ids):
    return OpenAIFalso([f"Jugador {i}" for i in ids])


def escribir_puntos(datos_dir, jornada, puntos):
    carpeta = datos_dir / str(TEMPORADA) / f"j{jornada:02d}"
    carpeta.mkdir(parents=True, exist_ok=True)
    contenido = {
        "puntos": [
            {"jugador_id": i, "nombre": f"Jugador {i}", "puntos": p} for i, p in puntos.items()
        ],
        "sin_asociar": [],
    }
    (carpeta / "puntos.json").write_text(json.dumps(contenido), encoding="utf-8")


def evaluar(datos_dir, plantilla=PLANTILLA, cliente=None, jornada=JORNADA):
    return evaluar_once(
        plantilla,
        jornada=jornada,
        temporada=TEMPORADA,
        datos_dir=datos_dir,
        indice=IndiceVacio(),
        modelo="modelo-de-prueba",
        cliente=cliente or cliente_con(ONCE_RECOMENDADO),
        esperar=lambda segundos: None,
    )


def puntos_simples(jugadores, **cambios):
    """Todos los de `jugadores` con 0 puntos salvo los indicados."""
    return {j["jugador_id"]: 0 for j in jugadores} | cambios


def test_el_mejor_once_posible_suma_los_puntos_de_la_mejor_formacion_valida(tmp_path):
    escribir_puntos(tmp_path, JORNADA, PUNTOS)

    assert evaluar(tmp_path)["mejor_once"]["puntos"] == 60


def test_el_mejor_once_es_un_1_4_4_2_si_los_centrocampistas_suman_mas(tmp_path):
    centrocampistas = {"c1": 9, "c2": 8, "c3": 7, "c4": 6}
    delanteros = {"f1": 10, "f2": 1, "f3": 0, "f4": 0}
    escribir_puntos(tmp_path, JORNADA, PUNTOS | centrocampistas | delanteros)

    mejor = evaluar(tmp_path)["mejor_once"]["jugadores"]

    assert Counter(POSICION[i] for i in mejor) == {
        "portero": 1,
        "defensa": 4,
        "centrocampista": 4,
        "delantero": 2,
    }


def test_los_puntos_del_once_recomendado_son_los_reales_de_sus_jugadores(tmp_path):
    escribir_puntos(tmp_path, JORNADA, PUNTOS)

    assert evaluar(tmp_path)["recomendado"]["puntos"] == 45


def test_la_proporcion_es_la_del_once_recomendado_frente_al_mejor_posible(tmp_path):
    escribir_puntos(tmp_path, JORNADA, PUNTOS)

    assert evaluar(tmp_path)["proporcion"] == pytest.approx(0.75)


def test_un_jugador_que_no_aparece_en_los_puntos_de_la_jornada_cuenta_con_cero(tmp_path):
    escribir_puntos(tmp_path, JORNADA, {i: p for i, p in PUNTOS.items() if i != "f2"})

    assert evaluar(tmp_path)["recomendado"]["puntos"] == 35


@pytest.mark.parametrize(
    "once, valido",
    [
        (ONCE_RECOMENDADO, True),
        (ONCE_RECOMENDADO[:10], False),
        ([*ONCE_RECOMENDADO[:10], "f1"], False),
        ([*ONCE_RECOMENDADO[:10], "ajeno"], False),
    ],
    ids=["correcto", "menos de 11", "repetido", "fuera de la plantilla"],
)
def test_el_resultado_avisa_si_el_once_recomendado_no_es_valido(tmp_path, once, valido):
    escribir_puntos(tmp_path, JORNADA, PUNTOS)

    assert evaluar(tmp_path, cliente=cliente_con(once))["recomendado"]["valido"] is valido


def test_un_once_no_valido_suma_solo_sus_jugadores_distintos_de_la_plantilla(tmp_path):
    escribir_puntos(tmp_path, JORNADA, PUNTOS)
    once = [*ONCE_RECOMENDADO[:9], "f2", "f2", "ajeno"]

    assert evaluar(tmp_path, cliente=cliente_con(once))["recomendado"]["puntos"] == 43


def test_en_la_jornada_1_la_eleccion_simple_es_nula_y_el_resto_se_calcula(tmp_path):
    escribir_puntos(tmp_path, 1, PUNTOS)

    resultado = evaluar(tmp_path, jornada=1)

    assert resultado["eleccion_simple"] is None
    assert resultado["mejor_once"]["puntos"] == 60


def test_la_eleccion_simple_usa_la_media_de_las_jornadas_anteriores(tmp_path):
    # Por media de j01 y j02 queda fuera f4. Solo con j02 quedaría fuera f1; contando la j03
    # evaluada, f2 o f3; contando la j04 posterior, f3.
    escribir_puntos(tmp_path, 1, puntos_simples(PLANTILLA_SIMPLE, f1=10, f2=3, f3=0, f4=1))
    escribir_puntos(tmp_path, 2, puntos_simples(PLANTILLA_SIMPLE, f1=0, f2=3, f3=4, f4=1))
    escribir_puntos(tmp_path, 3, puntos_simples(PLANTILLA_SIMPLE, f1=1, f2=2, f3=4, f4=8))
    escribir_puntos(tmp_path, 4, puntos_simples(PLANTILLA_SIMPLE, f4=30))

    resultado = evaluar(
        tmp_path, plantilla=PLANTILLA_SIMPLE, cliente=cliente_con(ONCE_SIMPLE), jornada=3
    )

    assert resultado["eleccion_simple"]["puntos"] == 7


def test_en_la_eleccion_simple_un_jugador_sin_puntos_anteriores_tiene_media_cero(tmp_path):
    previos = puntos_simples(PLANTILLA_SIMPLE, f1=5, f2=3, f3=-1)
    del previos["f4"]
    escribir_puntos(tmp_path, 1, previos)
    escribir_puntos(tmp_path, 2, puntos_simples(PLANTILLA_SIMPLE, f3=2, f4=7))

    resultado = evaluar(
        tmp_path, plantilla=PLANTILLA_SIMPLE, cliente=cliente_con(ONCE_SIMPLE), jornada=2
    )

    assert resultado["eleccion_simple"]["puntos"] == 7


def test_en_la_eleccion_simple_un_empate_lo_gana_el_que_va_antes_en_la_plantilla(tmp_path):
    # f3 y f4 empatan en media; f4 tiene nombre e identificador que irían antes al ordenar.
    plantilla = [*PLANTILLA_SIMPLE[:-1], jugador("a0", "delantero")]
    escribir_puntos(tmp_path, 1, puntos_simples(plantilla, f1=5, f2=3, f3=1, a0=1))
    escribir_puntos(tmp_path, 2, puntos_simples(plantilla, f3=2, a0=7))

    resultado = evaluar(tmp_path, plantilla=plantilla, cliente=cliente_con(ONCE_SIMPLE), jornada=2)

    assert resultado["eleccion_simple"]["puntos"] == 2


def escribir_jornadas_anteriores(datos_dir):
    escribir_puntos(datos_dir, 10, PUNTOS)
    escribir_puntos(datos_dir, 11, PUNTOS)


def test_sin_los_puntos_de_la_jornada_evaluada_el_error_nombra_jornada_y_temporada(tmp_path):
    escribir_jornadas_anteriores(tmp_path)

    with pytest.raises(Exception) as error:
        evaluar(tmp_path)

    assert "12" in str(error.value) and "2026" in str(error.value)


def test_sin_los_puntos_de_la_jornada_evaluada_no_se_pide_la_recomendacion(tmp_path):
    escribir_jornadas_anteriores(tmp_path)
    cliente = cliente_con(ONCE_RECOMENDADO)

    with pytest.raises(Exception) as error:
        evaluar(tmp_path, cliente=cliente)

    assert cliente.peticiones == [], error.value


def test_una_plantilla_sin_formacion_valida_da_error_sin_pedir_la_recomendacion(tmp_path):
    plantilla = plantilla_de(
        defensas=["d1", "d2", "d3", "d4", "d5"],
        centrocampistas=["c1", "c2", "c3"],
        delanteros=["f1", "f2", "f3"],
    )
    escribir_puntos(tmp_path, JORNADA, puntos_simples(plantilla))
    cliente = cliente_con([j["jugador_id"] for j in plantilla])

    with pytest.raises(ErrorPlantilla, match="formación"):
        evaluar(tmp_path, plantilla=plantilla, cliente=cliente)

    assert cliente.peticiones == []
