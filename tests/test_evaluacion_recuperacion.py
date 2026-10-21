import json
from types import SimpleNamespace

import pytest

from fantasy_copilot.evaluacion import ErrorJornadaSinIngerir, evaluar_recuperacion
from fantasy_copilot.indice import TOP_K, Indice

JORNADA = 12
TEMPORADA = 2026

PEDRI = {
    "nombre": "Pedri",
    "jugador_id": "683840",
    "equipo": "FC Barcelona",
    "posicion": "Mediocentro",
}
RAPHINHA = {
    "nombre": "Raphinha",
    "jugador_id": "411295",
    "equipo": "FC Barcelona",
    "posicion": "Extremo derecho",
}
OBLAK = {
    "nombre": "Jan Oblak",
    "jugador_id": "121483",
    "equipo": "Atlético de Madrid",
    "posicion": "Portero",
}


class OpenAIFalso:
    """Hace de cliente de OpenAI: da el mismo vector a todo texto y guarda los textos pedidos."""

    def __init__(self):
        self.textos = []
        self.embeddings = SimpleNamespace(create=self._crear)

    def _crear(self, *, model, input):
        self.textos.extend(input)
        return SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 0.0]) for _ in input])


class IndiceElegido:
    """Un `Indice` real cuya consulta devuelve, por jugador preguntado, los insights elegidos.

    `elegidos` asocia el nombre que aparece en la pregunta con los `video_id` de los insights
    que se recuperan, en orden y con repeticiones si las hay.
    """

    def __init__(self, elegidos):
        self.elegidos = elegidos
        self._real = Indice(cliente=OpenAIFalso(), modelo="modelo-de-prueba")

    def indexar(self, insights):
        self._real.indexar(insights)

    def consultar(self, pregunta, *, jornada, temporada, top_k=TOP_K):
        todos = self._real.consultar(pregunta, jornada=jornada, temporada=temporada, top_k=1000)
        por_url = {r["metadatos"]["url"]: r for r in todos}
        nombre = next((n for n in self.elegidos if n in pregunta), None)
        if nombre is None:
            return []
        return [por_url[url(video_id)] for video_id in self.elegidos[nombre]][:top_k]


def url(video_id):
    return f"https://youtube.com/watch?v={video_id}"


def insight(jugador, video_id, jornada=JORNADA, temporada=TEMPORADA):
    """Un insight normalizado. Dos insights del mismo jugador solo difieren en su vídeo."""
    return {
        "jugador": jugador["nombre"],
        "jugador_id": jugador["jugador_id"],
        "equipo": jugador["equipo"],
        "posicion": jugador["posicion"],
        "nombre_transcrito": jugador["nombre"],
        "accion": "alinear",
        "confianza": 0.9,
        "razon_corta": "En racha",
        "criterios": {
            "forma_actual": None,
            "dinamica_equipo": None,
            "calendario_rival": None,
            "prediccion_puntos": None,
        },
        "quote_original": None,
        "fuente": {
            "canal": "carrasco",
            "video_id": video_id,
            "titulo": f"Jornada {jornada}",
            "fecha": "2026-11-05",
            "jornada": jornada,
            "temporada": temporada,
            "url": url(video_id),
            "chunk": None,
        },
    }


def escribir_normalizados(datos_dir, insights, jornada=JORNADA, temporada=TEMPORADA):
    carpeta = datos_dir / str(temporada) / f"j{jornada:02d}"
    carpeta.mkdir(parents=True, exist_ok=True)
    (carpeta / "insights_normalizados.json").write_text(
        json.dumps(insights, ensure_ascii=False), encoding="utf-8"
    )


def evaluar(datos_dir, plantilla, indice):
    return evaluar_recuperacion(
        plantilla, jornada=JORNADA, temporada=TEMPORADA, datos_dir=datos_dir, indice=indice
    )


def del_jugador(resultado, jugador):
    [entrada] = [j for j in resultado["jugadores"] if j["jugador_id"] == jugador["jugador_id"]]
    return entrada


JORNADA_BASE = [
    insight(PEDRI, "pedri1"),
    insight(PEDRI, "pedri2"),
    insight(PEDRI, "pedri3"),
    insight(RAPHINHA, "raphinha1"),
    insight(RAPHINHA, "raphinha2"),
    insight(OBLAK, "oblak1"),
]
ELEGIDOS_BASE = {
    "Pedri": ["pedri1", "raphinha1", "pedri2", "raphinha2", "oblak1"],
    "Raphinha": ["raphinha1", "raphinha2"],
}


def test_precision_y_cobertura_de_un_jugador_con_parte_de_lo_recuperado_ajeno(tmp_path):
    escribir_normalizados(tmp_path, JORNADA_BASE)

    resultado = evaluar(tmp_path, [PEDRI], IndiceElegido(ELEGIDOS_BASE))

    pedri = del_jugador(resultado, PEDRI)
    assert (pedri["precision"], pedri["cobertura"]) == (pytest.approx(0.4), pytest.approx(2 / 3))


def test_las_medias_son_las_de_cada_jugador_y_no_las_del_total(tmp_path):
    escribir_normalizados(tmp_path, JORNADA_BASE)

    resultado = evaluar(tmp_path, [PEDRI, RAPHINHA], IndiceElegido(ELEGIDOS_BASE))

    assert (resultado["precision"], resultado["cobertura"]) == (
        pytest.approx(0.7),
        pytest.approx(5 / 6),
    )


def test_un_jugador_sin_insights_en_la_jornada_no_entra_en_las_medias(tmp_path):
    escribir_normalizados(tmp_path, JORNADA_BASE[:5])
    elegidos = {"Pedri": ELEGIDOS_BASE["Pedri"][:4] + ["pedri3"], "Oblak": ["pedri1"]}
    # Pedri: 5 recuperados, 3 suyos de 3 → precisión 0,6 y cobertura 1.

    resultado = evaluar(tmp_path, [PEDRI, OBLAK], IndiceElegido(elegidos))

    assert (resultado["precision"], resultado["cobertura"]) == (
        pytest.approx(0.6),
        pytest.approx(1.0),
    )


def test_un_jugador_sin_insights_en_la_jornada_aparece_como_sin_insights(tmp_path):
    escribir_normalizados(tmp_path, JORNADA_BASE[:5])

    elegidos = {"Pedri": ["pedri1", "raphinha1"], "Oblak": ["pedri1"]}

    resultado = evaluar(tmp_path, [PEDRI, OBLAK], IndiceElegido(elegidos))

    assert del_jugador(resultado, OBLAK)["sin_insights"] is True
    assert del_jugador(resultado, PEDRI)["sin_insights"] is False


def test_si_ningun_jugador_tiene_insights_las_medias_son_nulas(tmp_path):
    escribir_normalizados(tmp_path, [insight(PEDRI, "pedri1")])

    resultado = evaluar(tmp_path, [OBLAK], IndiceElegido({"Oblak": ["pedri1"]}))

    assert (resultado["precision"], resultado["cobertura"]) == (None, None)


def test_un_insight_recuperado_dos_veces_cuenta_una_y_dos_insights_distintos_cuentan_dos(
    tmp_path,
):
    escribir_normalizados(tmp_path, JORNADA_BASE[:3])

    resultado = evaluar(tmp_path, [PEDRI], IndiceElegido({"Pedri": ["pedri1", "pedri1"]}))

    assert del_jugador(resultado, PEDRI)["cobertura"] == pytest.approx(1 / 3)


def test_se_pregunta_al_indice_con_el_mismo_texto_que_la_recomendacion(tmp_path):
    escribir_normalizados(tmp_path, [insight(PEDRI, "pedri1")])
    openai_falso = OpenAIFalso()

    evaluar(tmp_path, [PEDRI], Indice(cliente=openai_falso, modelo="modelo-de-prueba"))

    assert "¿Qué dicen los expertos sobre Pedri en fantasy football?" in openai_falso.textos


def test_solo_se_recuperan_insights_de_la_jornada_y_temporada_evaluadas(tmp_path):
    escribir_normalizados(tmp_path, [insight(PEDRI, "pedri1")])
    indice = Indice(cliente=OpenAIFalso(), modelo="modelo-de-prueba")
    indice.indexar(
        [
            insight(PEDRI, "otra_jornada", jornada=JORNADA + 1),
            insight(PEDRI, "otra_temporada", temporada=TEMPORADA - 1),
        ]
    )

    resultado = evaluar(tmp_path, [PEDRI], indice)

    assert del_jugador(resultado, PEDRI)["precision"] == pytest.approx(1.0)


def test_se_recuperan_como_mucho_10_insights_por_jugador(tmp_path):
    escribir_normalizados(tmp_path, [insight(PEDRI, f"pedri{n:02d}") for n in range(12)])

    resultado = evaluar(tmp_path, [PEDRI], Indice(cliente=OpenAIFalso(), modelo="modelo-de-prueba"))

    assert del_jugador(resultado, PEDRI)["cobertura"] == pytest.approx(10 / 12)


def test_sin_insights_normalizados_se_lanza_un_error_que_pide_ejecutar_la_ingesta(tmp_path):
    with pytest.raises(ErrorJornadaSinIngerir) as error:
        evaluar(tmp_path, [PEDRI], IndiceElegido(ELEGIDOS_BASE))

    mensaje = str(error.value)
    assert "12" in mensaje and "2026" in mensaje and "ingesta" in mensaje.lower()
