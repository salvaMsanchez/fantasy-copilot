import json
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from fantasy_copilot.puntos import ErrorPuntos, obtener_puntos

MADRID = ZoneInfo("Europe/Madrid")
DESPUES_DE_LA_JORNADA = datetime(2026, 10, 20, 12, 0, tzinfo=MADRID)

ADRIAN_NINO = {
    "id": "100",
    "nombre": "Adrián Niño",
    "posicion": "Delantero centro",
    "equipo": "Málaga CF",
    "equipos": ["Málaga CF"],
}
ALEX_NINO = {
    "id": "101",
    "nombre": "Álex Niño",
    "posicion": "Lateral derecho",
    "equipo": "Málaga CF",
    "equipos": ["Málaga CF"],
}
ALFONSO_HERRERO = {
    "id": "200",
    "nombre": "Alfonso Herrero",
    "posicion": "Portero",
    "equipo": "Málaga CF",
    "equipos": ["Málaga CF"],
}
ISI_PALAZON = {
    "id": "300",
    "nombre": "Isi Palazón",
    "posicion": "Extremo derecho",
    "equipo": "Rayo Vallecano",
    "equipos": ["Rayo Vallecano"],
}
ARAMBARRI = {
    "id": "400",
    "nombre": "Mauro Arambarri",
    "posicion": "Pivote",
    "equipo": "Getafe CF",
    "equipos": ["Getafe CF"],
}
LISTA_OFICIAL = [ADRIAN_NINO, ALFONSO_HERRERO, ISI_PALAZON, ARAMBARRI]


def url_puntos(anio_fin, jornada):
    return f"https://www.futbolfantasy.com/laliga/puntos/{anio_fin}/{jornada}/laliga-fantasy"


URL_J8_2026 = url_puntos(2027, 8)


def fila(nombre_corto, ficha, puntos, comunio="4", comunio_primero=False, cambio=None):
    """`cambio` es `(tipo, minuto)`, p. ej. `("Entrada", "61'")`, si el jugador entra o sale."""
    caja = "racha-box columna_puntos mx-auto point"
    laliga = f'<span class="{caja} high laliga-fantasy ">{puntos}</span>'
    otro = f'<span class="{caja} low comunio scale2">{comunio}</span>'
    spans = otro + laliga if comunio_primero else laliga + otro
    icono = ""
    if cambio:
        tipo, minuto = cambio
        archivo = "icono_entra.png" if tipo == "Entrada" else "icono_sale.png"
        icono = (
            f'\n    <img title="{tipo}" alt="{tipo}" src="{archivo}">'
            f' <span class="minuto">{minuto}</span>'
        )
    return f"""
<tr class="plegado plegable">
  <td class="name" data-posicion-laliga-fantasy="Delantero">
    <i class="fa fa-plus desplegar" aria-hidden="true"></i>
    <i class="fa fa-minus plegar" aria-hidden="true"></i> {nombre_corto}{icono}
  </td>
  <td class="data bold d-flex">
    {spans}
  </td>
</tr>
<tr class="desglose">
  <td colspan="3">
    <div class="desg laliga-fantasy">
      <a class="link" href="https://www.futbolfantasy.com/jugadores/{ficha}/laliga-26-27">Ver la ficha del jugador</a>
    </div>
  </td>
</tr>"""  # noqa: E501


def bloque(equipo, *filas):
    return f"""
<h2 class="title">Puntos <span class="game"></span> del {equipo}</h2>
<table><tbody>{"".join(filas)}</tbody></table>"""


def partido(fecha, *bloques):
    return f"""
<div class="col-6 col-lg-2">
    <div class="local score">1</div>
    <div class="visitante score">1</div>
</div>
<div class="fecha">
    Jornada 8. {fecha} a las 21:00h
</div>
{"".join(bloques)}"""


def pagina(*filas_malaga):
    """Jornada 8 de 2026: el partido del lunes 12 aparece antes que el del viernes 9."""
    lunes = partido(
        "Lunes, 12 de octubre del 2026",
        bloque("Málaga", *filas_malaga),
        bloque("Rayo", fila("Isi", "isi-palazon", "5")),
    )
    viernes = partido(
        "Viernes, 09 de octubre del 2026",
        bloque("Getafe", fila("Arambarri", "mauro-arambarri", "3")),
        bloque("Elche"),
    )
    return f"<html><body>{lunes}{viernes}</body></html>"


def pagina_con(*bloques):
    return f"<html><body>{partido('Lunes, 12 de octubre del 2026', *bloques)}</body></html>"


def oficial(id_, nombre, equipo):
    return {
        "id": id_,
        "nombre": nombre,
        "posicion": "Mediocentro",
        "equipo": equipo,
        "equipos": [equipo],
    }


PAGINA_J8 = pagina(
    fila("Niño", "adrian-nino", "9"),
    fila("Herrero", "alfonso-herrero", "6", comunio="2", comunio_primero=True),
)
PAGINA_SIN_JUGADORES = "<html><body><p>La jornada aún no se ha disputado</p></body></html>"


@dataclass
class Respuesta:
    status_code: int
    text: str


class FutbolFantasyFalso:
    """Sustituye al cliente HTTP. Cada página es un HTML, un código o un error."""

    def __init__(self, paginas):
        self.paginas = paginas
        self.urls = []

    def get(self, url, headers=None, **_):
        self.urls.append(url)
        pagina = self.paginas.get(url, 404)
        if isinstance(pagina, Exception):
            raise pagina
        if isinstance(pagina, int):
            return Respuesta(pagina, "")
        return Respuesta(200, pagina)


def obtener(web, datos_dir, *, jornada=8, temporada=2026, jugadores=None, ahora=None):
    return obtener_puntos(
        jornada,
        temporada,
        jugadores=LISTA_OFICIAL if jugadores is None else jugadores,
        datos_dir=datos_dir,
        cliente=web,
        ahora=DESPUES_DE_LA_JORNADA if ahora is None else ahora,
    )


def por_id(resultado):
    return {p["jugador_id"]: p for p in resultado["puntos"]}


def ficheros(carpeta):
    return [ruta for ruta in carpeta.rglob("*") if ruta.is_file()]


@pytest.mark.parametrize(
    "jornada, temporada, url",
    [
        pytest.param(8, 2026, url_puntos(2027, 8), id="jornada_8_de_2026"),
        pytest.param(12, 2025, url_puntos(2026, 12), id="jornada_12_de_2025"),
    ],
)
def test_pide_una_sola_vez_la_pagina_de_la_jornada_con_el_anio_en_que_acaba_la_temporada(
    tmp_path, jornada, temporada, url
):
    web = FutbolFantasyFalso({url: PAGINA_J8})

    obtener(web, tmp_path, jornada=jornada, temporada=temporada)

    assert web.urls == [url]


def test_cada_jugador_queda_con_el_id_y_el_nombre_oficial_y_sus_puntos_de_laliga_fantasy(
    tmp_path,
):
    web = FutbolFantasyFalso({URL_J8_2026: PAGINA_J8})

    resultado = obtener(web, tmp_path)

    puntos = por_id(resultado)
    assert puntos["100"] == {"jugador_id": "100", "nombre": "Adrián Niño", "puntos": 9}
    assert puntos["200"] == {"jugador_id": "200", "nombre": "Alfonso Herrero", "puntos": 6}


def test_con_dos_jugadores_del_mismo_equipo_con_el_mismo_apellido_cada_ficha_va_a_su_jugador(
    tmp_path,
):
    html = pagina(fila("Niño", "adrian-nino", "9"), fila("Niño", "alex-nino", "1"))
    web = FutbolFantasyFalso({URL_J8_2026: html})

    resultado = obtener(web, tmp_path, jugadores=[ALEX_NINO, *LISTA_OFICIAL])

    puntos = por_id(resultado)
    assert (puntos["100"]["puntos"], puntos["101"]["puntos"]) == (9, 1)


@pytest.mark.parametrize(
    "equipo_pagina, ficha, jugador",
    [
        pytest.param(
            "Celta",
            "lvaro-nunez",
            oficial("600", "Álvaro Núñez", "RC Celta de Vigo"),
            id="pierde_la_letra_acentuada_inicial",
        ),
        pytest.param(
            "Real Madrid",
            "arda-gler",
            oficial("601", "Arda Güler", "Real Madrid CF"),
            id="pierde_una_letra_acentuada_interior",
        ),
        pytest.param(
            "Getafe",
            "jesus-martinez-1",
            oficial("602", "Jesús Martínez", "Getafe CF"),
            id="acaba_en_numero_de_homonimo",
        ),
    ],
)
def test_una_ficha_escrita_distinta_que_la_lista_oficial_se_asocia_a_su_jugador(
    tmp_path, equipo_pagina, ficha, jugador
):
    web = FutbolFantasyFalso(
        {URL_J8_2026: pagina_con(bloque(equipo_pagina, fila("X", ficha, "4")))}
    )

    resultado = obtener(web, tmp_path, jugadores=[*LISTA_OFICIAL, jugador])

    assert resultado["puntos"] == [
        {"jugador_id": jugador["id"], "nombre": jugador["nombre"], "puntos": 4}
    ]


def test_una_ficha_con_una_letra_no_acentuada_de_menos_queda_sin_asociar(tmp_path):
    web = FutbolFantasyFalso({URL_J8_2026: pagina_con(bloque("Betis", fila("Marc", "marc", "3")))})
    marco = oficial("700", "Marco", "Real Betis Balompié")

    resultado = obtener(web, tmp_path, jugadores=[*LISTA_OFICIAL, marco])

    assert (resultado["puntos"], resultado["sin_asociar"]) == (
        [],
        [{"nombre": "marc", "equipo": "Betis", "puntos": 3}],
    )


def test_una_ficha_con_un_nombre_de_mas_que_la_lista_oficial_se_asocia_a_su_jugador(tmp_path):
    html = pagina_con(
        bloque("Betis", fila("Isco", "isco-alarcon", "18")),
        bloque("Barcelona", fila("Pedri", "pedri-gonzalez", "7")),
    )
    web = FutbolFantasyFalso({URL_J8_2026: html})
    isco = oficial("900", "Isco", "Real Betis Balompié")
    pedri = oficial("901", "Pedri", "FC Barcelona")

    resultado = obtener(web, tmp_path, jugadores=[*LISTA_OFICIAL, isco, pedri])

    assert por_id(resultado) == {
        "900": {"jugador_id": "900", "nombre": "Isco", "puntos": 18},
        "901": {"jugador_id": "901", "nombre": "Pedri", "puntos": 7},
    }


def test_una_ficha_con_un_nombre_de_menos_que_la_lista_oficial_se_asocia_a_su_jugador(tmp_path):
    web = FutbolFantasyFalso(
        {URL_J8_2026: pagina_con(bloque("Getafe", fila("Terrats", "terrats", "5")))}
    )
    terrats = oficial("910", "Ramón Terrats", "Getafe CF")

    resultado = obtener(web, tmp_path, jugadores=[*LISTA_OFICIAL, terrats])

    assert resultado["puntos"] == [{"jugador_id": "910", "nombre": "Ramón Terrats", "puntos": 5}]


def test_una_ficha_que_contiene_el_nombre_oficial_no_se_asocia_si_el_nombre_corto_lo_contradice(
    tmp_path,
):
    web = FutbolFantasyFalso(
        {URL_J8_2026: pagina_con(bloque("Málaga", fila("Céspedes", "andres-cespedes", "4")))}
    )
    andres = oficial("920", "Andrés", "Málaga CF")

    resultado = obtener(web, tmp_path, jugadores=[*LISTA_OFICIAL, andres])

    assert (resultado["puntos"], resultado["sin_asociar"]) == (
        [],
        [{"nombre": "andres cespedes", "equipo": "Málaga", "puntos": 4}],
    )


def test_el_minuto_de_salida_no_impide_asociar_una_ficha_con_un_nombre_de_mas(tmp_path):
    html = pagina_con(bloque("Betis", fila("Isco", "isco-alarcon", "18", cambio=("Salida", "75'"))))
    web = FutbolFantasyFalso({URL_J8_2026: html})
    isco = oficial("900", "Isco", "Real Betis Balompié")

    resultado = obtener(web, tmp_path, jugadores=[*LISTA_OFICIAL, isco])

    assert resultado["puntos"] == [{"jugador_id": "900", "nombre": "Isco", "puntos": 18}]


def test_el_minuto_de_entrada_no_impide_asociar_una_ficha_con_un_nombre_de_menos(tmp_path):
    html = pagina_con(bloque("Getafe", fila("Terrats", "terrats", "5", cambio=("Entrada", "61'"))))
    web = FutbolFantasyFalso({URL_J8_2026: html})
    terrats = oficial("910", "Ramón Terrats", "Getafe CF")

    resultado = obtener(web, tmp_path, jugadores=[*LISTA_OFICIAL, terrats])

    assert resultado["puntos"] == [{"jugador_id": "910", "nombre": "Ramón Terrats", "puntos": 5}]


def test_con_minuto_de_entrada_el_nombre_corto_que_contradice_la_ficha_sigue_dejandola_sin_asociar(
    tmp_path,
):
    html = pagina_con(
        bloque("Málaga", fila("Céspedes", "andres-cespedes", "4", cambio=("Entrada", "46'")))
    )
    web = FutbolFantasyFalso({URL_J8_2026: html})
    andres = oficial("920", "Andrés", "Málaga CF")

    resultado = obtener(web, tmp_path, jugadores=[*LISTA_OFICIAL, andres])

    assert (resultado["puntos"], resultado["sin_asociar"]) == (
        [],
        [{"nombre": "andres cespedes", "equipo": "Málaga", "puntos": 4}],
    )


def test_una_ficha_con_un_nombre_de_menos_que_encaja_con_dos_jugadores_del_equipo_queda_sin_asociar(
    tmp_path,
):
    web = FutbolFantasyFalso(
        {URL_J8_2026: pagina_con(bloque("Getafe", fila("Terrats", "terrats", "5")))}
    )
    ramon = oficial("910", "Ramón Terrats", "Getafe CF")
    pau = oficial("911", "Pau Terrats", "Getafe CF")

    resultado = obtener(web, tmp_path, jugadores=[*LISTA_OFICIAL, ramon, pau])

    assert (resultado["puntos"], resultado["sin_asociar"]) == (
        [],
        [{"nombre": "terrats", "equipo": "Getafe", "puntos": 5}],
    )


def test_con_dos_equipos_que_encajan_con_el_nombre_corto_cada_bloque_va_a_su_equipo(tmp_path):
    html = pagina_con(
        bloque("Deportivo", fila("Lucas Pérez", "lucas-perez", "8")),
        bloque("Alavés", fila("Lucas Pérez", "lucas-perez", "2")),
    )
    web = FutbolFantasyFalso({URL_J8_2026: html})
    del_deportivo = oficial("800", "Lucas Pérez", "RC Deportivo A Coruña")
    del_alaves = oficial("801", "Lucas Pérez", "Deportivo Alavés")

    resultado = obtener(web, tmp_path, jugadores=[del_alaves, del_deportivo, *LISTA_OFICIAL])

    puntos = por_id(resultado)
    assert (puntos["800"]["puntos"], puntos["801"]["puntos"]) == (8, 2)


def test_los_puntos_negativos_se_conservan(tmp_path):
    web = FutbolFantasyFalso({URL_J8_2026: pagina(fila("Herrero", "alfonso-herrero", "-2"))})

    resultado = obtener(web, tmp_path)

    assert por_id(resultado)["200"]["puntos"] == -2


def test_un_convocado_que_no_jugo_queda_con_cero_puntos(tmp_path):
    web = FutbolFantasyFalso({URL_J8_2026: pagina(fila("Herrero", "alfonso-herrero", "-"))})

    resultado = obtener(web, tmp_path)

    assert por_id(resultado)["200"]["puntos"] == 0


def test_un_jugador_que_no_esta_en_su_equipo_de_la_lista_oficial_queda_aparte(tmp_path):
    html = pagina(fila("Niño", "adrian-nino", "9"), fila("Céspedes", "andres-cespedes", "-"))
    web = FutbolFantasyFalso({URL_J8_2026: html})
    cespedes_de_otro_equipo = {
        "id": "500",
        "nombre": "Andrés Céspedes",
        "posicion": "Pivote",
        "equipo": "Getafe CF",
        "equipos": ["Getafe CF"],
    }

    resultado = obtener(web, tmp_path, jugadores=[*LISTA_OFICIAL, cespedes_de_otro_equipo])

    assert resultado["sin_asociar"] == [
        {"nombre": "andres cespedes", "equipo": "Málaga", "puntos": 0}
    ]
    assert "500" not in por_id(resultado)


def test_guarda_las_dos_listas_en_el_fichero_de_la_jornada_y_nada_mas(tmp_path):
    web = FutbolFantasyFalso({URL_J8_2026: PAGINA_J8})

    resultado = obtener(web, tmp_path)

    fichero = tmp_path / "2026" / "j08" / "puntos.json"
    assert ficheros(tmp_path) == [fichero]
    guardado = json.loads(fichero.read_text(encoding="utf-8"))
    assert guardado == {"puntos": resultado["puntos"], "sin_asociar": resultado["sin_asociar"]}


def test_si_el_fichero_de_la_jornada_existe_devuelve_su_contenido_sin_peticiones(tmp_path):
    guardado = {
        "puntos": [{"jugador_id": "999", "nombre": "Ya Guardado", "puntos": 7}],
        "sin_asociar": [{"nombre": "otro guardado", "equipo": "Rayo", "puntos": 1}],
    }
    carpeta = tmp_path / "2026" / "j08"
    carpeta.mkdir(parents=True)
    (carpeta / "puntos.json").write_text(json.dumps(guardado), encoding="utf-8")
    web = FutbolFantasyFalso({URL_J8_2026: PAGINA_J8})

    resultado = obtener(web, tmp_path)

    assert (resultado["puntos"], resultado["sin_asociar"]) == (
        guardado["puntos"],
        guardado["sin_asociar"],
    )
    assert web.urls == []


def test_antes_de_tres_dias_desde_el_ultimo_partido_los_puntos_son_provisionales_y_no_se_guardan(
    tmp_path,
):
    web = FutbolFantasyFalso({URL_J8_2026: PAGINA_J8})

    resultado = obtener(web, tmp_path, ahora=datetime(2026, 10, 15, 20, 59, tzinfo=MADRID))

    assert resultado["provisional"] is True
    assert ficheros(tmp_path) == []


def test_a_los_tres_dias_del_ultimo_partido_los_puntos_son_definitivos_y_se_guardan(tmp_path):
    web = FutbolFantasyFalso({URL_J8_2026: PAGINA_J8})

    resultado = obtener(web, tmp_path, ahora=datetime(2026, 10, 15, 21, 0, tzinfo=MADRID))

    assert resultado["provisional"] is False
    assert (tmp_path / "2026" / "j08" / "puntos.json").exists()


@pytest.mark.parametrize(
    "jornada, pagina_web",
    [
        pytest.param(8, 500, id="codigo_distinto_de_200"),
        pytest.param(8, ConnectionError("sin conexión"), id="error_de_red"),
        pytest.param(12, PAGINA_SIN_JUGADORES, id="pagina_sin_jugadores"),
    ],
)
def test_si_no_se_obtienen_puntos_el_error_nombra_la_jornada_y_la_temporada(
    tmp_path, jornada, pagina_web
):
    web = FutbolFantasyFalso({url_puntos(2027, jornada): pagina_web})

    with pytest.raises(ErrorPuntos, match=rf"(?s)(?=.*\b{jornada}\b)(?=.*\b2026\b)"):
        obtener(web, tmp_path, jornada=jornada)


def test_si_la_pagina_no_tiene_jugadores_no_se_escribe_ningun_fichero(tmp_path):
    web = FutbolFantasyFalso({url_puntos(2027, 12): PAGINA_SIN_JUGADORES})

    with pytest.raises(ErrorPuntos):
        obtener(web, tmp_path, jornada=12)

    assert ficheros(tmp_path) == []
