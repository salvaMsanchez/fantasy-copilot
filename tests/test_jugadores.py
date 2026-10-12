import json
from dataclasses import dataclass

import pytest

from fantasy_copilot.jugadores import ErrorTransfermarkt, extraer_jugadores, obtener_jugadores

BASE = "https://www.transfermarkt.es"

BELLINGHAM = {
    "id": "581678",
    "nombre": "Jude Bellingham",
    "posicion": "Mediocentro ofensivo",
    "equipo": "Real Madrid",
}
MBAPPE = {
    "id": "342229",
    "nombre": "Kylian Mbappé",
    "posicion": "Delantero centro",
    "equipo": "Real Madrid",
}
ARAMBARRI = {
    "id": "256178",
    "nombre": "Mauro Arambarri",
    "posicion": "Pivote",
    "equipo": "Getafe CF",
}
LISTA_DE_LA_TEMPORADA = [
    {**BELLINGHAM, "equipos": ["Real Madrid"]},
    {**MBAPPE, "equipos": ["Real Madrid"]},
    {**ARAMBARRI, "equipos": ["Getafe CF"]},
]

PLANTILLA_REAL_MADRID = """
<table class="items"><tbody>
  <tr>
    <td class="posrela">
      <table class="inline-table">
        <tr><td rowspan="2"><img src="x.png"></td><td class="hauptlink"><a href="/jude-bellingham/profil/spieler/581678">Jude Bellingham</a></td></tr>
        <tr><td>Mediocentro ofensivo</td></tr>
      </table>
    </td>
  </tr>
  <tr>
    <td class="posrela">
      <table class="inline-table">
        <tr><td rowspan="2"><img src="y.png"></td><td class="hauptlink"><a href="/kylian-mbappe/profil/spieler/342229">Kylian Mbappé</a></td></tr>
        <tr><td>Delantero centro</td></tr>
      </table>
    </td>
  </tr>
</tbody></table>
"""  # noqa: E501


def fila(ruta, nombre, posicion="Pivote"):
    celda_posicion = "" if posicion is None else f"<tr><td>{posicion}</td></tr>"
    return f"""
  <tr>
    <td class="posrela">
      <table class="inline-table">
        <tr><td rowspan="2"><img src="x.png"></td><td class="hauptlink"><a href="{ruta}">{nombre}</a></td></tr>
        {celda_posicion}
      </table>
    </td>
  </tr>"""  # noqa: E501


def plantilla(*filas):
    return f'<table class="items"><tbody>{"".join(filas)}</tbody></table>'


PLANTILLA_GETAFE = plantilla(fila("/mauro-arambarri/profil/spieler/256178", "Mauro Arambarri"))


def url_liga(temporada):
    return f"{BASE}/laliga/startseite/wettbewerb/ES1/plus/?saison_id={temporada}"


def url_real_madrid(temporada):
    return f"{BASE}/real-madrid/kader/verein/418/saison_id/{temporada}"


def url_getafe(temporada):
    return f"{BASE}/getafe-cf/kader/verein/3709/saison_id/{temporada}"


def pagina_liga(temporada):
    return f"""
<table class="items"><tbody>
  <tr><td class="hauptlink"><a href="/real-madrid/startseite/verein/418/saison_id/{temporada}">Real Madrid</a></td></tr>
  <tr><td class="hauptlink"><a href="/getafe-cf/startseite/verein/3709/saison_id/{temporada}">Getafe CF</a></td></tr>
</tbody></table>
"""  # noqa: E501


def paginas(temporada):
    return {
        url_liga(temporada): pagina_liga(temporada),
        url_real_madrid(temporada): PLANTILLA_REAL_MADRID,
        url_getafe(temporada): PLANTILLA_GETAFE,
    }


@dataclass
class Respuesta:
    status_code: int
    text: str


class TransfermarktFalso:
    """Sustituye al cliente HTTP y a la espera. Cada página es un HTML, un código o un error."""

    def __init__(self, paginas):
        self.paginas = paginas
        self.eventos = []

    def get(self, url, headers=None, **_):
        self.eventos.append(("get", url, dict(headers or {})))
        pagina = self.paginas.get(url, 404)
        if isinstance(pagina, Exception):
            raise pagina
        if isinstance(pagina, int):
            return Respuesta(pagina, "")
        return Respuesta(200, pagina)

    def esperar(self, segundos):
        self.eventos.append(("esperar", segundos))

    @property
    def urls(self):
        return [evento[1] for evento in self.eventos if evento[0] == "get"]


def obtener(temporada, datos_dir, web):
    return obtener_jugadores(temporada, datos_dir=datos_dir, cliente=web, esperar=web.esperar)


def ficheros(carpeta):
    return [ruta for ruta in carpeta.rglob("*") if ruta.is_file()]


def test_de_la_pagina_de_plantilla_se_obtienen_los_jugadores_con_su_identificador():
    assert extraer_jugadores(PLANTILLA_REAL_MADRID, "Real Madrid") == [BELLINGHAM, MBAPPE]


def test_una_fila_de_cabecera_no_produce_ningun_jugador():
    html = plantilla(
        "<tr><th>Jugador</th></tr>",
        fila("/mauro-arambarri/profil/spieler/256178", "Mauro Arambarri"),
    )

    assert extraer_jugadores(html, "Getafe CF") == [ARAMBARRI]


def test_una_fila_cuyo_enlace_no_es_un_perfil_de_jugador_se_ignora():
    html = plantilla(
        fila("/getafe-cf/startseite/verein/3709", "Getafe CF"),
        fila("/mauro-arambarri/profil/spieler/256178", "Mauro Arambarri"),
    )

    assert extraer_jugadores(html, "Getafe CF") == [ARAMBARRI]


@pytest.mark.parametrize(
    "posicion",
    [pytest.param(None, id="sin_fila_de_posicion"), pytest.param("  ", id="celda_vacia")],
)
def test_un_jugador_sin_posicion_se_devuelve_con_posicion_nula(posicion):
    html = plantilla(fila("/mauro-arambarri/profil/spieler/256178", "Mauro Arambarri", posicion))

    assert extraer_jugadores(html, "Getafe CF") == [{**ARAMBARRI, "posicion": None}]


def test_dos_jugadores_con_el_mismo_nombre_y_distinto_identificador_se_conservan():
    html = plantilla(
        fila("/gonzalo-garcia/profil/spieler/111", "Gonzalo García"),
        fila("/gonzalo-garcia/profil/spieler/222", "Gonzalo García"),
    )

    jugadores = extraer_jugadores(html, "Real Madrid")

    assert [(j["id"], j["nombre"]) for j in jugadores] == [
        ("111", "Gonzalo García"),
        ("222", "Gonzalo García"),
    ]


def test_un_identificador_repetido_en_la_pagina_se_devuelve_una_sola_vez():
    html = plantilla(
        fila("/gonzalo-garcia/profil/spieler/111", "Gonzalo García"),
        fila("/gonzalo-garcia/profil/spieler/111", "Gonzalo García"),
    )

    jugadores = extraer_jugadores(html, "Real Madrid")

    assert [j["id"] for j in jugadores] == ["111"]


def test_devuelve_los_jugadores_de_todos_los_equipos_de_la_liga(tmp_path):
    web = TransfermarktFalso(paginas(2026))

    jugadores = obtener(2026, tmp_path / "datos", web)

    assert jugadores == LISTA_DE_LA_TEMPORADA


@pytest.mark.parametrize("temporada", [2025, 2026])
def test_la_pagina_de_la_liga_se_consulta_con_la_temporada_pedida(tmp_path, temporada):
    web = TransfermarktFalso(paginas(temporada))

    obtener(temporada, tmp_path / "datos", web)

    assert web.urls[0] == url_liga(temporada)


def test_guarda_la_lista_en_el_fichero_de_la_temporada(tmp_path):
    web = TransfermarktFalso(paginas(2026))

    obtener(2026, tmp_path / "datos", web)

    guardado = json.loads((tmp_path / "datos" / "jugadores_2026.json").read_text(encoding="utf-8"))
    assert guardado == LISTA_DE_LA_TEMPORADA


def test_si_el_fichero_de_la_temporada_existe_devuelve_su_contenido_sin_peticiones(tmp_path):
    guardado = [{"id": "999", "nombre": "Ya Guardado", "posicion": None, "equipo": "Getafe CF"}]
    (tmp_path / "jugadores_2026.json").write_text(json.dumps(guardado), encoding="utf-8")
    web = TransfermarktFalso(paginas(2026))

    jugadores = obtener(2026, tmp_path, web)

    assert jugadores == guardado
    assert web.urls == []


def test_si_la_pagina_de_un_equipo_responde_503_el_error_nombra_al_equipo(tmp_path):
    web = TransfermarktFalso({**paginas(2025), url_getafe(2025): 503})

    with pytest.raises(ErrorTransfermarkt, match="Getafe CF"):
        obtener(2025, tmp_path / "datos", web)


def test_si_la_pagina_de_un_equipo_da_error_de_red_el_error_nombra_al_equipo(tmp_path):
    web = TransfermarktFalso({**paginas(2025), url_getafe(2025): ConnectionError("sin conexión")})

    with pytest.raises(ErrorTransfermarkt, match="Getafe CF"):
        obtener(2025, tmp_path / "datos", web)


def test_si_la_pagina_de_la_liga_falla_el_error_nombra_la_liga(tmp_path):
    web = TransfermarktFalso({**paginas(2025), url_liga(2025): 500})

    with pytest.raises(ErrorTransfermarkt, match="(?i)laliga"):
        obtener(2025, tmp_path / "datos", web)


def test_si_falla_un_equipo_no_se_escribe_ningun_fichero(tmp_path):
    web = TransfermarktFalso({**paginas(2025), url_getafe(2025): 503})

    with pytest.raises(ErrorTransfermarkt):
        obtener(2025, tmp_path / "datos", web)

    assert ficheros(tmp_path) == []


LIGA_SIN_EQUIPOS = "<html><body>Acceso denegado</body></html>"
PLANTILLA_SIN_JUGADORES = '<table class="items"><tbody></tbody></table>'


def test_si_la_pagina_de_la_liga_no_contiene_equipos_el_error_nombra_la_liga(tmp_path):
    web = TransfermarktFalso({**paginas(2025), url_liga(2025): LIGA_SIN_EQUIPOS})

    with pytest.raises(ErrorTransfermarkt, match="LaLiga"):
        obtener(2025, tmp_path / "datos", web)


def test_si_la_plantilla_de_un_equipo_no_contiene_jugadores_el_error_nombra_al_equipo(tmp_path):
    web = TransfermarktFalso({**paginas(2025), url_getafe(2025): PLANTILLA_SIN_JUGADORES})

    with pytest.raises(ErrorTransfermarkt, match="Getafe CF"):
        obtener(2025, tmp_path / "datos", web)


@pytest.mark.parametrize(
    "url, html",
    [
        pytest.param(url_liga(2025), LIGA_SIN_EQUIPOS, id="liga_sin_equipos"),
        pytest.param(url_getafe(2025), PLANTILLA_SIN_JUGADORES, id="plantilla_sin_jugadores"),
    ],
)
def test_si_una_pagina_no_contiene_lo_que_se_busca_no_se_escribe_ningun_fichero(
    tmp_path, url, html
):
    web = TransfermarktFalso({**paginas(2025), url: html})

    with pytest.raises(ErrorTransfermarkt):
        obtener(2025, tmp_path / "datos", web)

    assert ficheros(tmp_path) == []


def test_todas_las_peticiones_llevan_user_agent_de_navegador(tmp_path):
    web = TransfermarktFalso(paginas(2026))

    obtener(2026, tmp_path / "datos", web)

    agentes = [
        {clave.lower(): valor for clave, valor in evento[2].items()}.get("user-agent", "")
        for evento in web.eventos
        if evento[0] == "get"
    ]
    assert len(agentes) == 3
    assert all("Mozilla" in agente for agente in agentes)


def test_entre_la_peticion_de_un_equipo_y_la_del_siguiente_se_espera_un_segundo(tmp_path):
    web = TransfermarktFalso(paginas(2026))

    obtener(2026, tmp_path / "datos", web)

    pasos = [evento[:2] for evento in web.eventos]
    primero = pasos.index(("get", url_real_madrid(2026)))
    segundo = pasos.index(("get", url_getafe(2026)))
    espera = sum(paso[1] for paso in pasos[primero:segundo] if paso[0] == "esperar")
    assert espera >= 1


RUTAS_DE_EQUIPO = {
    "FC Barcelona": "fc-barcelona/{}/verein/131",
    "Girona FC": "girona-fc/{}/verein/12321",
}
RUTA_TER_STEGEN = "/marc-ter-stegen/profil/spieler/74857"
FILA_TER_STEGEN = fila(RUTA_TER_STEGEN, "Marc ter Stegen", "Portero")


def web_con_plantillas(temporada, plantillas):
    """La página de la liga lista los equipos en el orden del diccionario `plantillas`."""
    enlaces, paginas_web = [], {}
    for equipo, html in plantillas.items():
        ruta = f"{RUTAS_DE_EQUIPO[equipo]}/saison_id/{temporada}"
        inicio = ruta.format("startseite")
        enlaces.append(f'<tr><td class="hauptlink"><a href="/{inicio}">{equipo}</a></td></tr>')
        paginas_web[f"{BASE}/{ruta.format('kader')}"] = html
    paginas_web[url_liga(temporada)] = (
        f'<table class="items"><tbody>{"".join(enlaces)}</tbody></table>'
    )
    return TransfermarktFalso(paginas_web)


def test_un_jugador_en_dos_plantillas_aparece_una_sola_vez_con_sus_dos_equipos(tmp_path):
    web = web_con_plantillas(
        2025,
        {"FC Barcelona": plantilla(FILA_TER_STEGEN), "Girona FC": plantilla(FILA_TER_STEGEN)},
    )

    jugadores = obtener(2025, tmp_path / "datos", web)

    assert jugadores == [
        {
            "id": "74857",
            "nombre": "Marc ter Stegen",
            "posicion": "Portero",
            "equipo": "FC Barcelona",
            "equipos": ["FC Barcelona", "Girona FC"],
        }
    ]


def test_los_equipos_de_un_jugador_siguen_el_orden_de_la_pagina_de_la_liga(tmp_path):
    web = web_con_plantillas(
        2025,
        {"Girona FC": plantilla(FILA_TER_STEGEN), "FC Barcelona": plantilla(FILA_TER_STEGEN)},
    )

    jugadores = obtener(2025, tmp_path / "datos", web)

    assert [(j["equipo"], j["equipos"]) for j in jugadores] == [
        ("Girona FC", ["Girona FC", "FC Barcelona"])
    ]


def test_el_jugador_repetido_ocupa_el_lugar_de_su_primera_aparicion(tmp_path):
    web = web_con_plantillas(
        2025,
        {
            "FC Barcelona": plantilla(FILA_TER_STEGEN, fila("/cien/profil/spieler/100", "Cien")),
            "Girona FC": plantilla(
                fila("/doscientos/profil/spieler/200", "Doscientos"),
                FILA_TER_STEGEN,
                fila("/trescientos/profil/spieler/300", "Trescientos"),
            ),
        },
    )

    jugadores = obtener(2025, tmp_path / "datos", web)

    assert [j["id"] for j in jugadores] == ["74857", "100", "200", "300"]


def test_el_nombre_y_la_posicion_de_un_jugador_repetido_son_los_de_su_primera_plantilla(tmp_path):
    web = web_con_plantillas(
        2025,
        {
            "FC Barcelona": plantilla(FILA_TER_STEGEN),
            "Girona FC": plantilla(fila(RUTA_TER_STEGEN, "M. ter Stegen", None)),
        },
    )

    jugadores = obtener(2025, tmp_path / "datos", web)

    assert [(j["nombre"], j["posicion"]) for j in jugadores] == [("Marc ter Stegen", "Portero")]


def test_el_fichero_guarda_al_jugador_repetido_una_sola_vez_igual_que_la_lista_devuelta(tmp_path):
    web = web_con_plantillas(
        2025,
        {"FC Barcelona": plantilla(FILA_TER_STEGEN), "Girona FC": plantilla(FILA_TER_STEGEN)},
    )

    jugadores = obtener(2025, tmp_path / "datos", web)

    guardado = json.loads((tmp_path / "datos" / "jugadores_2025.json").read_text(encoding="utf-8"))
    assert [j["id"] for j in guardado] == ["74857"]
    assert guardado == jugadores
