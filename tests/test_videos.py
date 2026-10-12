import pytest

from fantasy_copilot.videos import (
    ErrorYouTube,
    es_video_relevante,
    extraer_numero_jornada,
    obtener_videos,
)


@pytest.mark.parametrize(
    ("titulo", "jornada"),
    [
        ("ONCE IDEAL JORNADA 12", 12),
        ("jornada3: análisis", 3),
        ("J12 onces probables", 12),
        ("Fichajes J7 | Fantasy", 7),
    ],
)
def test_la_jornada_se_reconoce_en_sus_tres_formas_sin_distinguir_mayusculas(titulo, jornada):
    assert extraer_numero_jornada(titulo) == jornada


def test_un_titulo_sin_numero_de_jornada_no_indica_ninguna():
    assert extraer_numero_jornada("Mis favoritos de la temporada") is None


def test_la_forma_abreviada_no_cuenta_si_la_j_no_empieza_palabra():
    assert extraer_numero_jornada("Lo mejor de MJ23") is None


@pytest.mark.parametrize("titulo", ["Jornada 45", "J0", "J39 resumen"])
def test_un_numero_fuera_de_1_a_38_no_indica_ninguna_jornada(titulo):
    assert extraer_numero_jornada(titulo) is None


def test_la_jornada_38_es_valida():
    assert extraer_numero_jornada("J38 final de liga") == 38


def test_un_titulo_con_jornada_abreviada_es_relevante():
    assert es_video_relevante("J12 onces probables") is True


def test_un_titulo_sin_jornada_no_es_relevante():
    assert es_video_relevante("Mis favoritos de la temporada") is False


def test_un_titulo_con_jornada_fuera_de_rango_no_es_relevante():
    assert es_video_relevante("J39 resumen") is False


@pytest.mark.parametrize("titulo", ["Mundial: jornada 2", "MUNDIAL J3"])
def test_un_titulo_del_mundial_no_es_relevante_aunque_indique_jornada(titulo):
    assert es_video_relevante(titulo) is False


CARRASCO = "UCu7ZOzCUgWpvvJzYAdnw0_Q"
JORNADA_PERFECTA = "UCvMw41zGXyYYPDvyXcHZfug"
NOMBRES = {CARRASCO: "carrasco", JORNADA_PERFECTA: "jornada_perfecta"}
MAXIMO_DE_PETICIONES = 20


def subidas(canal_id):
    return "UU" + canal_id[2:]


def video(video_id, titulo, publicado):
    return {
        "snippet": {
            "title": titulo,
            "publishedAt": publicado,
            "resourceId": {"videoId": video_id},
        }
    }


class FalloDeRed(Exception):
    pass


class _Peticion:
    def __init__(self, responder):
        self._responder = responder

    def execute(self):
        return self._responder()


class _Recurso:
    def __init__(self, responder):
        self._responder = responder

    def list(self, **parametros):
        return _Peticion(lambda: self._responder(**parametros))


class YouTubeFalso:
    """Sustituye al cliente de YouTube Data API. Cada canal es una lista de páginas de vídeos.

    `fallos` son pares (nombre del canal, llamada) cuya respuesta es un error.
    """

    def __init__(self, paginas_por_canal, fallos=()):
        self.paginas = {
            CARRASCO: paginas_por_canal.get("carrasco", [[]]),
            JORNADA_PERFECTA: paginas_por_canal.get("jornada_perfecta", [[]]),
        }
        self.fallos = set(fallos)
        self.paginas_pedidas = []

    def channels(self):
        return _Recurso(self._canal)

    def playlistItems(self):
        return _Recurso(self._pagina)

    def _canal(self, id, part, **_):
        if (NOMBRES[id], "channels") in self.fallos:
            raise FalloDeRed("quota exceeded")
        return {"items": [{"contentDetails": {"relatedPlaylists": {"uploads": subidas(id)}}}]}

    def _pagina(self, playlistId, part, pageToken=None, **_):
        canal_id = next(canal for canal in self.paginas if subidas(canal) == playlistId)
        nombre = NOMBRES[canal_id]
        if (nombre, "playlistItems") in self.fallos:
            raise FalloDeRed("quota exceeded")
        paginas = self.paginas[canal_id]
        indice = int(pageToken.removeprefix("pagina-")) if pageToken else 0
        self.paginas_pedidas.append((nombre, indice))
        assert len(self.paginas_pedidas) <= MAXIMO_DE_PETICIONES, "se piden páginas sin fin"
        respuesta = {"items": paginas[indice]}
        if indice + 1 < len(paginas):
            respuesta["nextPageToken"] = f"pagina-{indice + 1}"
        return respuesta


def identificadores(videos):
    return [v["video_id"] for v in videos]


def test_de_un_elemento_de_youtube_se_obtiene_el_video_con_fecha_jornada_y_direccion():
    youtube = YouTubeFalso(
        {"carrasco": [[video("AT_UTIeNmtI", "Jornada 12", "2026-11-05T18:00:00Z")]]}
    )

    assert obtener_videos(12, 2026, cliente=youtube) == [
        {
            "canal": "carrasco",
            "video_id": "AT_UTIeNmtI",
            "titulo": "Jornada 12",
            "fecha": "2026-11-05",
            "jornada": 12,
            "temporada": 2026,
            "url": "https://youtube.com/watch?v=AT_UTIeNmtI",
        }
    ]


def test_sin_videos_de_esa_jornada_y_temporada_se_devuelve_una_lista_vacia():
    youtube = YouTubeFalso(
        {
            "carrasco": [[video("otra", "Jornada 11", "2026-10-29T18:00:00Z")]],
            "jornada_perfecta": [[]],
        }
    )

    assert obtener_videos(12, 2026, cliente=youtube) == []


def test_solo_se_devuelven_los_videos_de_la_jornada_y_la_temporada_pedidas():
    youtube = YouTubeFalso(
        {
            "carrasco": [
                [
                    video("j13", "Jornada 13", "2026-11-12T18:00:00Z"),
                    video("j12", "Jornada 12", "2026-11-05T18:00:00Z"),
                    video("j12-anterior", "Jornada 12", "2025-11-08T18:00:00Z"),
                ]
            ]
        }
    )

    assert identificadores(obtener_videos(12, 2026, cliente=youtube)) == ["j12"]


def test_la_temporada_va_del_1_de_julio_al_30_de_junio_ambos_incluidos():
    youtube = YouTubeFalso(
        {
            "carrasco": [
                [
                    video("despues", "Jornada 12", "2027-07-01T12:00:00Z"),
                    video("ultimo-dia", "Jornada 12", "2027-06-30T12:00:00Z"),
                    video("primer-dia", "Jornada 12", "2026-07-01T12:00:00Z"),
                    video("antes", "Jornada 12", "2026-06-30T12:00:00Z"),
                ]
            ]
        }
    )

    assert identificadores(obtener_videos(12, 2026, cliente=youtube)) == [
        "ultimo-dia",
        "primer-dia",
    ]


def test_un_video_de_la_segunda_pagina_de_un_canal_se_devuelve():
    youtube = YouTubeFalso(
        {
            "carrasco": [
                [
                    video("j14", "Jornada 14", "2026-11-19T18:00:00Z"),
                    video("j13", "Jornada 13", "2026-11-12T18:00:00Z"),
                ],
                [video("j12", "Jornada 12", "2026-11-05T18:00:00Z")],
            ]
        }
    )

    assert identificadores(obtener_videos(12, 2026, cliente=youtube)) == ["j12"]


def test_tras_una_pagina_sin_videos_con_next_page_token_se_pide_la_siguiente():
    youtube = YouTubeFalso(
        {
            "carrasco": [
                [],
                [video("j12", "Jornada 12", "2026-11-05T18:00:00Z")],
            ]
        }
    )

    assert identificadores(obtener_videos(12, 2026, cliente=youtube)) == ["j12"]


def test_se_retrocede_mas_alla_de_una_pagina_entera_posterior_a_la_temporada():
    youtube = YouTubeFalso(
        {
            "carrasco": [
                [video("j12-siguiente", "Jornada 12", "2026-11-05T18:00:00Z")],
                [video("j12", "Jornada 12", "2025-11-08T18:00:00Z")],
            ]
        }
    )

    assert identificadores(obtener_videos(12, 2025, cliente=youtube)) == ["j12"]


def test_sin_next_page_token_no_se_pide_otra_pagina():
    youtube = YouTubeFalso(
        {
            "carrasco": [[video("j12", "Jornada 12", "2026-11-05T18:00:00Z")]],
            "jornada_perfecta": [[video("j12-jp", "Jornada 12", "2026-11-06T18:00:00Z")]],
        }
    )

    obtener_videos(12, 2026, cliente=youtube)

    assert sorted(youtube.paginas_pedidas) == [("carrasco", 0), ("jornada_perfecta", 0)]


def test_no_se_pide_otra_pagina_si_toda_la_recibida_es_anterior_a_la_temporada():
    youtube = YouTubeFalso(
        {
            "carrasco": [
                [video("j12", "Jornada 12", "2026-11-05T18:00:00Z")],
                [
                    video("j38", "Jornada 38", "2026-05-21T18:00:00Z"),
                    video("j37", "Jornada 37", "2026-05-14T18:00:00Z"),
                ],
                [video("j36", "Jornada 36", "2026-05-07T18:00:00Z")],
            ]
        }
    )

    obtener_videos(12, 2026, cliente=youtube)

    assert [p for p in youtube.paginas_pedidas if p[0] == "carrasco"] == [
        ("carrasco", 0),
        ("carrasco", 1),
    ]


def test_se_reunen_los_videos_de_todos_los_canales_del_mas_reciente_al_mas_antiguo():
    youtube = YouTubeFalso(
        {
            "carrasco": [
                [
                    video("carrasco-7", "J12 últimas noticias", "2026-11-07T18:00:00Z"),
                    video("carrasco-5", "Jornada 12", "2026-11-05T18:00:00Z"),
                ]
            ],
            "jornada_perfecta": [[video("jp-6", "Jornada 12", "2026-11-06T18:00:00Z")]],
        }
    )

    videos = obtener_videos(12, 2026, cliente=youtube)

    assert [(v["canal"], v["video_id"]) for v in videos] == [
        ("carrasco", "carrasco-7"),
        ("jornada_perfecta", "jp-6"),
        ("carrasco", "carrasco-5"),
    ]


@pytest.mark.parametrize("llamada", ["channels", "playlistItems"])
def test_si_falla_la_consulta_de_un_canal_el_error_lo_nombra(llamada):
    youtube = YouTubeFalso(
        {"carrasco": [[video("j12", "Jornada 12", "2026-11-05T18:00:00Z")]]},
        fallos={("jornada_perfecta", llamada)},
    )

    with pytest.raises(ErrorYouTube, match="jornada_perfecta"):
        obtener_videos(12, 2026, cliente=youtube)
