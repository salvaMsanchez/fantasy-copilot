import pytest

from fantasy_copilot.videos import es_video_relevante, extraer_numero_jornada


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
