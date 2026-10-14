import json
import re

import pytest

from fantasy_copilot.plantilla import ErrorPlantilla, cargar_plantilla


def oficial(id, nombre, posicion, equipo):
    return {"id": id, "nombre": nombre, "posicion": posicion, "equipo": equipo, "equipos": [equipo]}


JUGADORES_OFICIALES = [
    oficial("1", "Augusto Batalla", "Portero", "Rayo Vallecano"),
    oficial("2", "Andrei Ratiu", "Lateral derecho", "Rayo Vallecano"),
    oficial("3", "Florian Lejeune", "Defensa central", "Rayo Vallecano"),
    oficial("4", "Pep Chavarría", "Lateral izquierdo", "Rayo Vallecano"),
    oficial("5", "Pathé Ciss", "Pivote", "Rayo Vallecano"),
    oficial("6", "Óscar Valentín", "Pivote", "Rayo Vallecano"),
    oficial("7", "Isi Palazón", "Extremo derecho", "Rayo Vallecano"),
    oficial("8", "Jorge de Frutos", "Extremo izquierdo", "Rayo Vallecano"),
    oficial("9", "Gonçalo Guedes", "Extremo izquierdo", "Real Sociedad"),
    oficial("10", "Vedat Muriqi", "Delantero centro", "RCD Mallorca"),
    oficial("11", "Alemão", "Delantero centro", "Rayo Vallecano"),
    oficial("12", "Gonzalo García", "Delantero centro", "Real Madrid"),
    oficial("13", "Gonzalo García", "Mediocentro", "Real Zaragoza"),
    oficial("14", "Unai López", "Mediocentro", "Rayo Vallecano"),
]

PLANTILLA_VALIDA = [
    {"nombre": "Augusto Batalla", "posicion": "portero"},
    {"nombre": "Andrei Ratiu", "posicion": "defensa"},
    {"nombre": "Florian Lejeune", "posicion": "defensa"},
    {"nombre": "Pep Chavarría", "posicion": "defensa"},
    {"nombre": "Pathé Ciss", "posicion": "centrocampista"},
    {"nombre": "Óscar Valentín", "posicion": "centrocampista"},
    {"nombre": "Isi Palazón", "posicion": "centrocampista"},
    {"nombre": "Jorge de Frutos", "posicion": "delantero"},
    {"nombre": "Gonçalo Guedes", "posicion": "delantero"},
    {"nombre": "Vedat Muriqi", "posicion": "delantero"},
    {"nombre": "Alemão", "posicion": "delantero"},
]


def escribir(tmp_path, plantilla):
    ruta = tmp_path / "plantilla.json"
    ruta.write_text(json.dumps(plantilla, ensure_ascii=False), encoding="utf-8")
    return ruta


def cargar(tmp_path, plantilla):
    return cargar_plantilla(escribir(tmp_path, plantilla), JUGADORES_OFICIALES)


def mensaje_de_error(tmp_path, plantilla):
    ruta = escribir(tmp_path, plantilla)
    with pytest.raises(ErrorPlantilla) as error:
        cargar_plantilla(ruta, JUGADORES_OFICIALES)
    return str(error.value).replace(str(ruta), "")


def test_una_plantilla_valida_devuelve_cada_jugador_oficial_con_la_posicion_del_fichero(tmp_path):
    plantilla = cargar(tmp_path, PLANTILLA_VALIDA)

    assert plantilla[:2] == [
        {
            "nombre": "Augusto Batalla",
            "jugador_id": "1",
            "equipo": "Rayo Vallecano",
            "posicion": "portero",
        },
        {
            "nombre": "Andrei Ratiu",
            "jugador_id": "2",
            "equipo": "Rayo Vallecano",
            "posicion": "defensa",
        },
    ]
    assert [j["jugador_id"] for j in plantilla] == [str(i) for i in range(1, 12)]


@pytest.mark.parametrize(
    "escrito, jugador_id, nombre_oficial",
    [
        pytest.param("goncalo guedes", "9", "Gonçalo Guedes", id="sin_tildes_ni_mayusculas"),
        pytest.param("VEDAT MURIQI", "10", "Vedat Muriqi", id="en_mayusculas"),
    ],
)
def test_el_nombre_se_asocia_sin_distinguir_mayusculas_ni_diacriticos(
    tmp_path, escrito, jugador_id, nombre_oficial
):
    entradas = [
        {**entrada, "nombre": escrito} if entrada["nombre"] == nombre_oficial else entrada
        for entrada in PLANTILLA_VALIDA
    ]

    plantilla = cargar(tmp_path, entradas)

    asociado = next(j for j in plantilla if j["jugador_id"] == jugador_id)
    assert asociado["nombre"] == nombre_oficial


def test_un_nombre_que_no_es_de_ningun_jugador_oficial_produce_un_error_que_lo_nombra(tmp_path):
    entradas = [*PLANTILLA_VALIDA, {"nombre": "Jugador Inventado", "posicion": "delantero"}]

    assert "Jugador Inventado" in mensaje_de_error(tmp_path, entradas)


def test_un_nombre_de_dos_jugadores_oficiales_produce_un_error_de_ambiguedad(tmp_path):
    entradas = [*PLANTILLA_VALIDA, {"nombre": "Gonzalo García", "posicion": "delantero"}]

    mensaje = mensaje_de_error(tmp_path, entradas)

    assert "Gonzalo García" in mensaje
    assert re.search(r"(?i)ambig[uü]", mensaje)


def test_un_jugador_repetido_en_el_fichero_produce_un_error_que_lo_nombra(tmp_path):
    entradas = [*PLANTILLA_VALIDA, {"nombre": "Andrei Ratiu", "posicion": "defensa"}]

    assert "Andrei Ratiu" in mensaje_de_error(tmp_path, entradas)


def test_un_jugador_repetido_con_otra_escritura_tambien_produce_error(tmp_path):
    entradas = [*PLANTILLA_VALIDA, {"nombre": "ANDREI RATIU", "posicion": "centrocampista"}]

    assert "andrei ratiu" in mensaje_de_error(tmp_path, entradas).lower()


def test_una_posicion_no_valida_produce_un_error_con_el_jugador_y_la_posicion(tmp_path):
    entradas = [
        {**entrada, "posicion": "extremo"} if entrada["nombre"] == "Isi Palazón" else entrada
        for entrada in PLANTILLA_VALIDA
    ]

    mensaje = mensaje_de_error(tmp_path, entradas)

    assert "Isi Palazón" in mensaje
    assert "extremo" in mensaje


def test_una_plantilla_de_menos_de_once_jugadores_indica_cuantos_tiene_y_cuantos_faltan(tmp_path):
    mensaje = mensaje_de_error(tmp_path, PLANTILLA_VALIDA[:10])

    assert re.search(r"\b10\b", mensaje)
    assert re.search(r"\b11\b", mensaje)
