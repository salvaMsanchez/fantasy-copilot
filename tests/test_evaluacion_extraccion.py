import json
from types import SimpleNamespace

import openai
import pytest

from fantasy_copilot.evaluacion import ErrorConjunto, evaluar_extraccion

VIDEO_ID = "abc123def45"


def texto(marca, esperados, chunk=0, video_id=VIDEO_ID, jornada=8):
    """Un texto del conjunto. `marca` lo distingue en la petición al modelo."""
    return {
        "canal": "jornada_perfecta",
        "video_id": video_id,
        "titulo": f"ONCE IDEAL JORNADA {jornada}",
        "fecha": "2026-10-16",
        "jornada": jornada,
        "temporada": 2026,
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "chunk": chunk,
        "texto": f"El analista repasa sus recomendaciones {marca}",
        "insights_esperados": [{"jugador": j, "accion": a} for j, a in esperados],
    }


def insight(jugador, accion):
    return {
        "jugador": jugador,
        "accion": accion,
        "confianza": 0.8,
        "razon_corta": "Buen momento",
        "criterios": {
            "forma_actual": None,
            "dinamica_equipo": None,
            "calendario_rival": None,
            "prediccion_puntos": None,
        },
        "quote_original": None,
    }


def completa(*pares):
    contenido = json.dumps({"insights": [insight(j, a) for j, a in pares]}, ensure_ascii=False)
    mensaje = SimpleNamespace(role="assistant", content=contenido, refusal=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=mensaje, finish_reason="stop")])


class OpenAIFalso:
    """Hace de cliente de OpenAI: responde a `chat.completions.create`.

    `reglas` es una lista de pares (marca, respuesta). Se aplica la primera regla cuya marca
    aparece en los mensajes de la petición. Si la respuesta es una excepción, se lanza.
    """

    def __init__(self, reglas):
        self.reglas = reglas
        self.peticiones = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._crear))

    def _crear(self, **peticion):
        enviado = json.dumps(peticion.get("messages"), ensure_ascii=False, default=str)
        self.peticiones.append(enviado)
        respuesta = next((r for marca, r in self.reglas if marca in enviado), None)
        if respuesta is None:
            raise AssertionError("Petición inesperada: ninguna regla coincide con su texto")
        if isinstance(respuesta, Exception):
            raise respuesta
        return respuesta


def escribir_conjunto(datos_dir, textos):
    fichero = datos_dir / "evaluacion" / "extraccion.json"
    fichero.parent.mkdir(parents=True, exist_ok=True)
    fichero.write_text(json.dumps({"textos": textos}, ensure_ascii=False), encoding="utf-8")


def evaluar(datos_dir, cliente):
    return evaluar_extraccion(
        datos_dir=datos_dir,
        modelo="modelo-de-prueba",
        cliente=cliente,
        esperar=lambda segundos: None,
    )


def recuento(resultado):
    return resultado["correctos"], resultado["omitidos"], resultado["inventados"]


def test_con_todo_lo_esperado_extraido_y_un_texto_vacio_las_metricas_son_perfectas(tmp_path):
    escribir_conjunto(
        tmp_path,
        [
            texto(
                "TEXTO_A",
                [("Rafiña", "alinear"), ("Pedri", "evitar"), ("Lamine", "alinear")],
                chunk=0,
            ),
            texto("TEXTO_B", [], chunk=1),
        ],
    )
    openai_falso = OpenAIFalso(
        [
            (
                "TEXTO_A",
                completa(("Rafiña", "alinear"), ("Pedri", "evitar"), ("Lamine", "alinear")),
            ),
            ("TEXTO_B", completa()),
        ]
    )

    resultado = evaluar(tmp_path, openai_falso)

    assert (*recuento(resultado), resultado["precision"], resultado["cobertura"]) == (
        3,
        0,
        0,
        1,
        1,
    )


def test_correctos_omitidos_inventados_precision_y_cobertura_con_aciertos_y_errores(tmp_path):
    escribir_conjunto(tmp_path, [texto("TEXTO_A", [("Rafiña", "alinear"), ("Pedri", "evitar")])])
    openai_falso = OpenAIFalso(
        [("TEXTO_A", completa(("rafiña", "alinear"), ("Lamine", "alinear")))]
    )

    resultado = evaluar(tmp_path, openai_falso)

    assert (*recuento(resultado), resultado["precision"], resultado["cobertura"]) == (
        1,
        1,
        1,
        0.5,
        0.5,
    )


NOMBRES_EQUIVALENTES = {
    "tildes": ("Raphinha Dias Belloli", "Ráphinhá Díás Bélloli"),
    "espacios de más": ("Raphinha Dias Belloli", "  Raphinha   Dias Belloli "),
}


@pytest.mark.parametrize(
    "esperado, extraido", NOMBRES_EQUIVALENTES.values(), ids=NOMBRES_EQUIVALENTES.keys()
)
def test_un_nombre_que_solo_difiere_en_tildes_o_espacios_cuenta_como_correcto(
    tmp_path, esperado, extraido
):
    escribir_conjunto(tmp_path, [texto("TEXTO_A", [(esperado, "alinear")])])
    openai_falso = OpenAIFalso([("TEXTO_A", completa((extraido, "alinear")))])

    resultado = evaluar(tmp_path, openai_falso)

    assert recuento(resultado) == (1, 0, 0)


def test_el_jugador_esperado_con_la_otra_accion_cuenta_como_omitido_e_inventado(tmp_path):
    escribir_conjunto(tmp_path, [texto("TEXTO_A", [("Pedri", "alinear")])])
    openai_falso = OpenAIFalso([("TEXTO_A", completa(("Pedri", "evitar")))])

    resultado = evaluar(tmp_path, openai_falso)

    assert recuento(resultado) == (0, 1, 1)


def test_un_insight_esperado_una_vez_y_extraido_dos_cuenta_un_correcto_y_un_inventado(tmp_path):
    escribir_conjunto(tmp_path, [texto("TEXTO_A", [("Pedri", "alinear")])])
    openai_falso = OpenAIFalso([("TEXTO_A", completa(("Pedri", "alinear"), ("Pedri", "alinear")))])

    resultado = evaluar(tmp_path, openai_falso)

    assert recuento(resultado) == (1, 0, 1)


def test_un_insight_esperado_dos_veces_y_extraido_una_cuenta_un_correcto_y_un_omitido(tmp_path):
    escribir_conjunto(tmp_path, [texto("TEXTO_A", [("Pedri", "alinear"), ("Pedri", "alinear")])])
    openai_falso = OpenAIFalso([("TEXTO_A", completa(("Pedri", "alinear")))])

    resultado = evaluar(tmp_path, openai_falso)

    assert recuento(resultado) == (1, 1, 0)


def test_el_resultado_detalla_los_recuentos_de_cada_texto(tmp_path):
    escribir_conjunto(
        tmp_path,
        [
            texto("TEXTO_A", [("Rafiña", "alinear"), ("Pedri", "evitar")], chunk=0),
            texto("TEXTO_B", [("Lamine", "alinear")], chunk=1),
        ],
    )
    openai_falso = OpenAIFalso(
        [
            ("TEXTO_A", completa(("Rafiña", "alinear"))),
            ("TEXTO_B", completa(("Lamine", "alinear"), ("Gavi", "evitar"))),
        ]
    )

    resultado = evaluar(tmp_path, openai_falso)

    assert resultado["textos"] == [
        {
            "video_id": VIDEO_ID,
            "chunk": 0,
            "correctos": 1,
            "omitidos": 1,
            "inventados": 0,
            "fallido": False,
        },
        {
            "video_id": VIDEO_ID,
            "chunk": 1,
            "correctos": 1,
            "omitidos": 0,
            "inventados": 1,
            "fallido": False,
        },
    ]


def test_un_texto_cuya_extraccion_falla_cuenta_como_fallido_y_sus_esperados_como_omitidos(
    tmp_path,
):
    escribir_conjunto(
        tmp_path,
        [
            texto("TEXTO_A", [("Rafiña", "alinear"), ("Pedri", "evitar")], chunk=0),
            texto("TEXTO_B", [("Lamine", "alinear")], chunk=1),
        ],
    )
    openai_falso = OpenAIFalso(
        [
            ("TEXTO_A", openai.OpenAIError("fallo de conexión")),
            ("TEXTO_B", completa(("Lamine", "alinear"))),
        ]
    )

    resultado = evaluar(tmp_path, openai_falso)

    assert (resultado["fallidos"], *recuento(resultado)) == (1, 1, 2, 0)


def guardar_insights_procesados(datos_dir):
    """Un insights.json de la J8 que ya da por procesado TEXTO_A con otro insight."""
    directorio = datos_dir / "2026" / "j08"
    directorio.mkdir(parents=True)
    fuente = {
        k: v for k, v in texto("TEXTO_A", []).items() if k not in ("texto", "insights_esperados")
    }
    (directorio / "insights.json").write_text(
        json.dumps(
            {
                "insights": [{**insight("Gavi", "evitar"), "fuente": fuente}],
                "procesados": [{"video_id": VIDEO_ID, "chunk": 0}],
                "fallidos": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_se_pide_al_modelo_cada_texto_aunque_la_jornada_ya_lo_tenga_procesado(tmp_path):
    guardar_insights_procesados(tmp_path)
    escribir_conjunto(tmp_path, [texto("TEXTO_A", [("Rafiña", "alinear")], chunk=0)])
    openai_falso = OpenAIFalso([("TEXTO_A", completa(("Rafiña", "alinear")))])

    resultado = evaluar(tmp_path, openai_falso)

    assert recuento(resultado) == (1, 0, 0)


def contenido_fuera_de_evaluacion(datos_dir):
    return {
        str(ruta.relative_to(datos_dir)): ruta.read_bytes() if ruta.is_file() else None
        for ruta in datos_dir.rglob("*")
        if ruta.relative_to(datos_dir).parts[0] != "evaluacion"
    }


def test_evaluar_no_escribe_nada_en_las_carpetas_de_las_jornadas(tmp_path):
    guardar_insights_procesados(tmp_path)
    escribir_conjunto(
        tmp_path,
        [
            texto("TEXTO_A", [("Rafiña", "alinear")], chunk=0, jornada=8),
            texto("TEXTO_B", [("Pedri", "evitar")], chunk=0, video_id="xyz987uvw65", jornada=9),
        ],
    )
    antes = contenido_fuera_de_evaluacion(tmp_path)
    openai_falso = OpenAIFalso(
        [
            ("TEXTO_A", completa(("Rafiña", "alinear"))),
            ("TEXTO_B", openai.OpenAIError("fallo de conexión")),
        ]
    )

    evaluar(tmp_path, openai_falso)

    assert contenido_fuera_de_evaluacion(tmp_path) == antes


def escribir_crudo(datos_dir, contenido):
    fichero = datos_dir / "evaluacion" / "extraccion.json"
    fichero.parent.mkdir(parents=True, exist_ok=True)
    fichero.write_text(contenido, encoding="utf-8")


def texto_sin(campo):
    return {k: v for k, v in texto("TEXTO_A", [("Rafiña", "alinear")]).items() if k != campo}


CONJUNTOS_INVALIDOS = {
    "no es JSON": '{"textos": [',
    "sin textos": json.dumps({"jornada": 8}),
    "textos no es una lista": json.dumps({"textos": {"a": 1}}),
    "texto sin insights_esperados": json.dumps({"textos": [texto_sin("insights_esperados")]}),
    "texto sin texto": json.dumps({"textos": [texto_sin("texto")]}),
    "texto sin video_id": json.dumps({"textos": [texto_sin("video_id")]}),
    "texto sin chunk": json.dumps({"textos": [texto_sin("chunk")]}),
    "texto sin jornada": json.dumps({"textos": [texto_sin("jornada")]}),
}


@pytest.mark.parametrize("contenido", CONJUNTOS_INVALIDOS.values(), ids=CONJUNTOS_INVALIDOS.keys())
def test_un_conjunto_mal_formado_lanza_un_error_que_nombra_el_fichero(tmp_path, contenido):
    escribir_crudo(tmp_path, contenido)

    with pytest.raises(ErrorConjunto, match="extraccion.json"):
        evaluar(tmp_path, OpenAIFalso([("TEXTO_A", completa(("Rafiña", "alinear")))]))


def test_si_el_conjunto_no_existe_se_lanza_un_error_que_nombra_el_fichero(tmp_path):
    with pytest.raises(ErrorConjunto, match="extraccion.json"):
        evaluar(tmp_path, OpenAIFalso([]))
