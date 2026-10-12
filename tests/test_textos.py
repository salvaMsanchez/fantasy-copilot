from itertools import pairwise

import pytest

from fantasy_copilot.textos import limpiar_texto, preparar_textos

VIDEO = {
    "canal": "carrasco",
    "video_id": "AT_UTIeNmtI",
    "titulo": "Jornada 12",
    "fecha": "2026-11-05",
    "jornada": 12,
    "temporada": 2026,
    "url": "https://youtube.com/watch?v=AT_UTIeNmtI",
}

DATOS_DEL_TROZO = ("texto", "chunk", "total_chunks")


def transcripcion(texto):
    return {
        **VIDEO,
        "estado": "ok",
        "motivo": None,
        "transcripcion_cruda": texto,
        "fragmentos": [{"texto": texto, "inicio": 0.0, "duracion": 2.5}],
    }


def texto_de(longitud):
    """Texto ya limpio de `longitud` caracteres exactos: `p00001 p00002 …`, sin puntuación."""
    palabras = [f"p{n:05d}" for n in range(1, longitud // 7 + 1)]
    texto = " ".join(palabras)
    return texto + "x" * (longitud - len(texto))


def solape(anterior, actual):
    """Longitud del final de `anterior` que se repite al principio de `actual`."""
    for longitud in range(min(len(anterior), len(actual)), 0, -1):
        if anterior.endswith(actual[:longitud]):
            return longitud
    return 0


def trozos_de(texto):
    return [t["texto"] for t in preparar_textos([transcripcion(texto)])]


def test_la_limpieza_conserva_el_apostrofo_y_el_guion_dentro_de_una_palabra():
    assert limpiar_texto("N'Diaye y Sidi-Mohamed") == "N'Diaye y Sidi-Mohamed"


@pytest.mark.parametrize("texto", ["Pedri - Lamine", "Pedri ' Lamine"])
def test_la_limpieza_elimina_el_apostrofo_y_el_guion_sueltos(texto):
    assert limpiar_texto(texto) == "Pedri Lamine"


def test_la_limpieza_conserva_tildes_enie_cifras_puntuacion_y_simbolos_de_cifras():
    texto = "¿Sube un 12%? Vale 7,5M€, +10 puntos; ¡ojo, Iñaki!"

    assert limpiar_texto(texto) == texto


def test_la_limpieza_conserva_la_dieresis_y_los_dos_puntos():
    assert limpiar_texto("Ojo: Güler") == "Ojo: Güler"


def test_la_limpieza_une_la_letra_y_su_acento_combinante_en_una_sola_letra():
    assert limpiar_texto("éxito de Vinícius") == "éxito de Vinícius"


def test_la_limpieza_elimina_las_etiquetas_entre_corchetes():
    assert limpiar_texto("hola [Música] a todos [Aplausos]") == "hola a todos"


def test_la_limpieza_elimina_una_etiqueta_con_un_salto_de_linea_dentro():
    assert limpiar_texto("hola [Risas y\naplausos] a todos") == "hola a todos"


def test_la_limpieza_elimina_otros_simbolos_y_deja_un_solo_espacio_entre_palabras():
    assert limpiar_texto("gol ♪♪ de Pedri >> ahora\n\nsí") == "gol de Pedri ahora sí"


def test_un_texto_limpio_de_500_caracteres_produce_un_texto():
    assert trozos_de(texto_de(500)) == [texto_de(500)]


def test_una_transcripcion_que_tras_limpiar_no_llega_a_500_caracteres_no_produce_texto():
    cruda = texto_de(499) + " [Música] [Aplausos]"

    assert preparar_textos([transcripcion(cruda)]) == []


def test_un_video_sin_transcripcion_disponible_no_produce_texto():
    no_disponible = {
        **VIDEO,
        "estado": "no_disponible",
        "motivo": "NoTranscriptFound",
        "transcripcion_cruda": None,
        "fragmentos": [],
    }

    assert preparar_textos([no_disponible]) == []


def test_el_texto_que_se_produce_es_el_limpio():
    cruda = "[Música]  " + texto_de(600) + " ♪♪"

    assert trozos_de(cruda) == [texto_de(600)]


def test_un_texto_limpio_de_15000_caracteres_produce_un_unico_texto_con_los_datos_del_video():
    texto = texto_de(15_000)

    assert preparar_textos([transcripcion(texto)]) == [
        {**VIDEO, "texto": texto, "chunk": None, "total_chunks": 1}
    ]


def test_un_texto_limpio_de_mas_de_15000_caracteres_se_divide_en_trozos_de_10000_como_maximo():
    trozos = trozos_de(texto_de(15_001))

    assert len(trozos) > 1
    assert max(len(trozo) for trozo in trozos) <= 10_000


def test_los_trozos_se_numeran_desde_1_y_llevan_el_total():
    textos = preparar_textos([transcripcion(texto_de(21_000))])

    total = len(textos)
    assert total > 1
    assert [(t["chunk"], t["total_chunks"]) for t in textos] == [
        (numero, total) for numero in range(1, total + 1)
    ]


def test_cada_trozo_lleva_los_datos_del_video_de_origen():
    textos = preparar_textos([transcripcion(texto_de(21_000))])

    assert len(textos) > 1
    datos = [
        {clave: valor for clave, valor in t.items() if clave not in DATOS_DEL_TROZO} for t in textos
    ]
    assert datos == [VIDEO] * len(textos)


def test_ningun_trozo_empieza_ni_termina_a_mitad_de_una_palabra():
    texto = texto_de(21_000)
    palabras = set(texto.split())

    trozos = trozos_de(texto)

    assert len(trozos) > 1
    extremos = [palabra for trozo in trozos for palabra in (trozo.split()[0], trozo.split()[-1])]
    assert [palabra for palabra in extremos if palabra not in palabras] == []


def test_cada_trozo_empieza_repitiendo_entre_300_y_500_caracteres_del_anterior():
    trozos = trozos_de(texto_de(21_000))

    assert len(trozos) > 1
    solapes = [solape(anterior, actual) for anterior, actual in pairwise(trozos)]
    assert [medida for medida in solapes if not 300 <= medida <= 500] == []


def test_los_trozos_sin_la_parte_repetida_reconstruyen_el_texto_limpio():
    texto = texto_de(21_000)

    trozos = trozos_de(texto)

    assert len(trozos) > 1
    partes = [trozos[0]] + [
        actual[solape(anterior, actual) :].strip() for anterior, actual in pairwise(trozos)
    ]
    assert " ".join(partes) == texto
