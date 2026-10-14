from types import SimpleNamespace

import pytest

from fantasy_copilot.configuracion import cargar_configuracion
from fantasy_copilot.indice import ErrorInsightNoNormalizado, Indice

RAPHINHA = {
    "jugador": "Raphinha",
    "jugador_id": "411295",
    "equipo": "FC Barcelona",
    "posicion": "Extremo derecho",
    "nombre_transcrito": "Rafiña",
}
LAMINE = {
    "jugador": "Lamine Yamal",
    "jugador_id": "937958",
    "equipo": "FC Barcelona",
    "posicion": "Extremo derecho",
    "nombre_transcrito": "la mine",
}
OBLAK = {
    "jugador": "Jan Oblak",
    "jugador_id": "121483",
    "equipo": "Atlético de Madrid",
    "posicion": "Portero",
    "nombre_transcrito": "Oblak",
}
PEDRI = {
    "jugador": "Pedri",
    "jugador_id": "683840",
    "equipo": "FC Barcelona",
    "posicion": "Mediocentro",
    "nombre_transcrito": "Pedri",
}
RAPHINHA_SIN_POSICION = {**RAPHINHA, "posicion": None}

# Vector fijo según el jugador que aparece en el texto. Lamine está más cerca de Raphinha
# que Oblak, así que una pregunta sobre Raphinha los ordena Raphinha, Lamine, Oblak.
VECTORES = {
    "Raphinha": [1.0, 0.0, 0.0, 0.0],
    "Lamine Yamal": [0.6, 0.8, 0.0, 0.0],
    "Jan Oblak": [0.0, 0.0, 1.0, 0.0],
}
VECTOR_POR_DEFECTO = [0.0, 0.0, 0.0, 1.0]


class OpenAIFalso:
    """Hace de cliente de OpenAI: responde a `embeddings.create` con vectores fijos."""

    def __init__(self):
        self.peticiones = []
        self.embeddings = SimpleNamespace(create=self._crear)

    def _crear(self, *, model, input):
        self.peticiones.append({"modelo": model, "textos": list(input)})
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=self._vector(texto)) for texto in input]
        )

    @staticmethod
    def _vector(texto):
        return next(
            (vector for nombre, vector in VECTORES.items() if nombre in texto),
            VECTOR_POR_DEFECTO,
        )


def normalizado(jugador=RAPHINHA, jornada=12, temporada=2026, video_id="AT_UTIeNmtI"):
    """Un insight con el formato de `normalizar_insights`."""
    return {
        "jugador": jugador["jugador"],
        "accion": "alinear",
        "confianza": 0.9,
        "razon_corta": "En racha y con rival asequible",
        "criterios": {
            "forma_actual": "Tres goles en dos partidos",
            "dinamica_equipo": None,
            "calendario_rival": None,
            "prediccion_puntos": None,
        },
        "quote_original": f"{jugador['nombre_transcrito']} es fijo, lo dicen en {video_id}",
        "fuente": {
            "canal": "carrasco",
            "video_id": video_id,
            "titulo": f"Jornada {jornada}",
            "fecha": "2026-11-05",
            "jornada": jornada,
            "temporada": temporada,
            "url": f"https://youtube.com/watch?v={video_id}",
            "chunk": None,
        },
        **jugador,
    }


def nuevo_indice(cliente=None, modelo="modelo-de-prueba"):
    return Indice(cliente=cliente or OpenAIFalso(), modelo=modelo)


def test_el_texto_indexado_lleva_el_nombre_oficial_y_no_el_oido():
    indice = nuevo_indice()
    indice.indexar([normalizado(RAPHINHA)])

    [resultado] = indice.consultar("Raphinha", jornada=12, temporada=2026)

    assert "Raphinha" in resultado["texto"]
    assert "Rafiña" not in resultado["texto"]


def test_los_metadatos_son_los_del_insight_normalizado():
    indice = nuevo_indice()
    indice.indexar([normalizado(RAPHINHA)])

    [resultado] = indice.consultar("Raphinha", jornada=12, temporada=2026)

    assert resultado["metadatos"] == {
        "jugador": "Raphinha",
        "jugador_id": "411295",
        "equipo": "FC Barcelona",
        "posicion": "Extremo derecho",
        "accion": "alinear",
        "confianza": 0.9,
        "canal": "carrasco",
        "fecha": "2026-11-05",
        "jornada": 12,
        "temporada": 2026,
        "url": "https://youtube.com/watch?v=AT_UTIeNmtI",
    }


def test_un_jugador_sin_posicion_se_recupera_con_los_11_metadatos_y_posicion_vacia():
    indice = nuevo_indice()
    indice.indexar([normalizado(RAPHINHA_SIN_POSICION)])

    [resultado] = indice.consultar("Raphinha", jornada=12, temporada=2026)

    assert resultado["metadatos"] == {
        "jugador": "Raphinha",
        "jugador_id": "411295",
        "equipo": "FC Barcelona",
        "posicion": "",
        "accion": "alinear",
        "confianza": 0.9,
        "canal": "carrasco",
        "fecha": "2026-11-05",
        "jornada": 12,
        "temporada": 2026,
        "url": "https://youtube.com/watch?v=AT_UTIeNmtI",
    }


def test_el_texto_de_un_jugador_sin_posicion_empieza_por_su_nombre_sin_mencionar_none():
    indice = nuevo_indice()
    indice.indexar([normalizado(RAPHINHA_SIN_POSICION)])

    [resultado] = indice.consultar("Raphinha", jornada=12, temporada=2026)

    assert resultado["texto"].startswith("Raphinha.")
    assert "None" not in resultado["texto"]


def test_el_texto_de_un_jugador_con_posicion_empieza_por_su_nombre_y_su_posicion():
    indice = nuevo_indice()
    indice.indexar([normalizado(PEDRI)])

    [resultado] = indice.consultar("Pedri", jornada=12, temporada=2026)

    assert resultado["texto"].startswith("Pedri, Mediocentro.")


def test_un_insight_sin_jugador_id_produce_un_error():
    sin_id = normalizado(RAPHINHA)
    del sin_id["jugador_id"]

    with pytest.raises(ErrorInsightNoNormalizado):
        nuevo_indice().indexar([sin_id])


def test_si_un_insight_no_tiene_jugador_id_no_se_indexa_ninguno_de_la_llamada():
    indice = nuevo_indice()
    sin_id = normalizado(LAMINE)
    del sin_id["jugador_id"]

    with pytest.raises(ErrorInsightNoNormalizado):
        indice.indexar([normalizado(RAPHINHA), sin_id])

    assert indice.consultar("Raphinha", jornada=12, temporada=2026) == []


def test_la_consulta_devuelve_top_k_resultados_del_mas_al_menos_parecido():
    indice = nuevo_indice()
    indice.indexar([normalizado(OBLAK), normalizado(LAMINE), normalizado(RAPHINHA)])

    resultados = indice.consultar("Raphinha", jornada=12, temporada=2026, top_k=2)

    assert [r["metadatos"]["jugador"] for r in resultados] == ["Raphinha", "Lamine Yamal"]
    assert resultados[0]["distancia"] < resultados[1]["distancia"]


def test_sin_top_k_la_consulta_devuelve_como_maximo_10_resultados():
    indice = nuevo_indice()
    indice.indexar([normalizado(video_id=f"video{n:02d}") for n in range(12)])

    resultados = indice.consultar("Raphinha", jornada=12, temporada=2026)

    assert len(resultados) == 10


def test_indexar_otra_jornada_conserva_los_insights_de_la_anterior():
    indice = nuevo_indice()
    indice.indexar([normalizado(RAPHINHA, jornada=12), normalizado(LAMINE, jornada=12)])
    indice.indexar([normalizado(RAPHINHA, jornada=13), normalizado(LAMINE, jornada=13)])

    resultados = indice.consultar("Raphinha", jornada=12, temporada=2026)

    assert sorted(r["metadatos"]["jugador"] for r in resultados) == ["Lamine Yamal", "Raphinha"]


def test_volver_a_indexar_una_jornada_no_duplica_entradas():
    indice = nuevo_indice()
    jornada_12 = [normalizado(RAPHINHA), normalizado(LAMINE)]
    indice.indexar(jornada_12)
    indice.indexar(jornada_12)

    resultados = indice.consultar("Raphinha", jornada=12, temporada=2026)

    assert len(resultados) == 2


def test_la_consulta_solo_devuelve_insights_de_su_jornada_y_temporada():
    indice = nuevo_indice()
    indice.indexar(
        [
            normalizado(RAPHINHA, jornada=12, temporada=2026),
            normalizado(RAPHINHA, jornada=13, temporada=2026),
            normalizado(RAPHINHA, jornada=12, temporada=2025),
        ]
    )

    resultados = indice.consultar("Raphinha", jornada=12, temporada=2026)

    assert [(r["metadatos"]["jornada"], r["metadatos"]["temporada"]) for r in resultados] == [
        (12, 2026)
    ]


def test_consultar_una_jornada_sin_insights_devuelve_una_lista_vacia():
    indice = nuevo_indice()
    indice.indexar([normalizado(RAPHINHA, jornada=12)])

    assert indice.consultar("Raphinha", jornada=14, temporada=2026) == []


def test_los_embeddings_se_piden_en_lotes_de_100_como_maximo():
    openai_falso = OpenAIFalso()
    indice = nuevo_indice(openai_falso)

    indice.indexar([normalizado(video_id=f"video{n:03d}") for n in range(250)])

    assert len(openai_falso.peticiones) == 3
    assert max(len(peticion["textos"]) for peticion in openai_falso.peticiones) <= 100


def test_los_embeddings_usan_el_modelo_de_la_configuracion(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FANTASY_MODELO_EMBEDDINGS", "modelo-e")
    openai_falso = OpenAIFalso()
    indice = nuevo_indice(openai_falso, modelo=cargar_configuracion().modelo_embeddings)

    indice.indexar([normalizado(RAPHINHA)])
    indice.consultar("Raphinha", jornada=12, temporada=2026)

    assert {peticion["modelo"] for peticion in openai_falso.peticiones} == {"modelo-e"}
