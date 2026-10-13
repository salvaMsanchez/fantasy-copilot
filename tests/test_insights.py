import json
import re
from types import SimpleNamespace

import openai
import pytest

from fantasy_copilot.configuracion import cargar_configuracion
from fantasy_copilot.insights import extraer_insights

VIDEO_ID = "AT_UTIeNmtI"

SIN_CRITERIOS = {
    "forma_actual": None,
    "dinamica_equipo": None,
    "calendario_rival": None,
    "prediccion_puntos": None,
}

INSIGHT_RAFINA = {
    "jugador": "Rafiña",
    "accion": "alinear",
    "confianza": 0.9,
    "razon_corta": "En racha y con rival asequible",
    "criterios": {
        "forma_actual": "Tres goles en dos partidos",
        "dinamica_equipo": None,
        "calendario_rival": "Recibe al colista",
        "prediccion_puntos": None,
    },
    "quote_original": "Rafiña es fijo esta jornada",
}

INSIGHT_LAMINE = {
    "jugador": "la mine",
    "accion": "evitar",
    "confianza": 0.6,
    "razon_corta": "Duda por molestias",
    "criterios": SIN_CRITERIOS,
    "quote_original": None,
}

PALABRAS = [f"p{n:05d}" for n in range(1, 201)]
TEXTO_LARGO = " ".join(PALABRAS)
TEXTO_ENTERO = (PALABRAS[0], PALABRAS[-1])
PRIMERA_MITAD = PALABRAS[0]
SEGUNDA_MITAD = PALABRAS[-1]


def texto(marca, chunk=None, video_id=VIDEO_ID, jornada=12, contenido=None):
    """Un texto con el formato de `preparar_textos`. `marca` lo distingue en la petición."""
    return {
        "canal": "carrasco",
        "video_id": video_id,
        "titulo": f"Jornada {jornada}",
        "fecha": "2026-11-05",
        "jornada": jornada,
        "temporada": 2026,
        "url": f"https://youtube.com/watch?v={video_id}",
        "texto": contenido or f"El analista repasa sus recomendaciones {marca}",
        "chunk": chunk,
        "total_chunks": 1 if chunk is None else 2,
    }


def fuente(chunk=None, video_id=VIDEO_ID, jornada=12):
    return {
        "canal": "carrasco",
        "video_id": video_id,
        "titulo": f"Jornada {jornada}",
        "fecha": "2026-11-05",
        "jornada": jornada,
        "temporada": 2026,
        "url": f"https://youtube.com/watch?v={video_id}",
        "chunk": chunk,
    }


def insight(jugador, **cambios):
    return {**INSIGHT_LAMINE, "jugador": jugador, **cambios}


def _respuesta(contenido, motivo_fin):
    mensaje = SimpleNamespace(role="assistant", content=contenido, refusal=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=mensaje, finish_reason=motivo_fin)])


def completa(*insights):
    return _respuesta(json.dumps({"insights": list(insights)}, ensure_ascii=False), "stop")


def cortada(contenido='{"insights": [{"jugador": "Raf'):
    return _respuesta(contenido, "length")


def error_api():
    return openai.OpenAIError("fallo de conexión")


def rechazada():
    mensaje = SimpleNamespace(role="assistant", content=None, refusal="No puedo ayudar con eso")
    return SimpleNamespace(choices=[SimpleNamespace(message=mensaje, finish_reason="stop")])


def a_medias():
    return _respuesta('{"insights": [', "stop")


class OpenAIFalso:
    """Hace de cliente de OpenAI: responde a `chat.completions.create`.

    `reglas` es una lista de pares (marcas, respuesta). Se aplica la primera regla cuyas marcas
    aparecen todas en los mensajes de la petición. La respuesta puede ser una excepción, que se
    lanza, o una lista, de la que se consume un elemento por petición y se repite el último.
    """

    def __init__(self, reglas):
        self.reglas = [
            ((marcas,) if isinstance(marcas, str) else marcas, respuesta)
            for marcas, respuesta in reglas
        ]
        self.peticiones = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._crear))

    def _crear(self, **peticion):
        enviado = json.dumps(peticion.get("messages"), ensure_ascii=False, default=str)
        self.peticiones.append({"modelo": peticion.get("model"), "enviado": enviado})
        respuesta = next(
            (r for marcas, r in self.reglas if all(marca in enviado for marca in marcas)), None
        )
        if respuesta is None:
            raise AssertionError("Petición inesperada: ninguna regla coincide con su texto")
        if isinstance(respuesta, list):
            respuesta = respuesta.pop(0) if len(respuesta) > 1 else respuesta[0]
        if isinstance(respuesta, Exception):
            raise respuesta
        return respuesta


def extraer(textos, datos_dir, cliente, jornada=12, modelo="modelo-de-prueba"):
    return extraer_insights(
        textos,
        jornada=jornada,
        temporada=2026,
        datos_dir=datos_dir,
        modelo=modelo,
        cliente=cliente,
        esperar=lambda segundos: None,
    )


def jugadores(resultado):
    return sorted(i["jugador"] for i in resultado["insights"])


def palabras_enteras_en(enviado):
    return [palabra for palabra in PALABRAS if re.search(rf"\b{palabra}\b", enviado)]


def fichero(datos_dir, jornada=12):
    return datos_dir / "2026" / f"j{jornada:02d}" / "insights.json"


def guardar(datos_dir, contenido, jornada=12):
    destino = fichero(datos_dir, jornada)
    destino.parent.mkdir(parents=True)
    destino.write_text(json.dumps(contenido, ensure_ascii=False), encoding="utf-8")


def test_cada_insight_de_la_respuesta_se_devuelve_con_sus_campos_y_la_fuente_del_texto(tmp_path):
    openai_falso = OpenAIFalso([("TEXTO_A", completa(INSIGHT_RAFINA, INSIGHT_LAMINE))])

    resultado = extraer([texto("TEXTO_A", chunk=1)], tmp_path, openai_falso)

    assert resultado["insights"] == [
        {**INSIGHT_RAFINA, "fuente": fuente(chunk=1)},
        {**INSIGHT_LAMINE, "fuente": fuente(chunk=1)},
    ]


def test_una_respuesta_con_lista_vacia_deja_el_texto_procesado_y_sin_insights(tmp_path):
    openai_falso = OpenAIFalso([("TEXTO_A", completa())])

    resultado = extraer([texto("TEXTO_A")], tmp_path, openai_falso)

    assert resultado == {
        "insights": [],
        "procesados": [{"video_id": VIDEO_ID, "chunk": None}],
        "fallidos": [],
    }


def test_un_texto_que_da_lista_vacia_se_consulta_una_sola_vez(tmp_path):
    openai_falso = OpenAIFalso([("TEXTO_A", completa())])

    extraer([texto("TEXTO_A")], tmp_path, openai_falso)

    assert len(openai_falso.peticiones) == 1


@pytest.mark.parametrize(
    "cambio",
    [{"accion": "fichar"}, {"confianza": 1.4}, {"confianza": -0.1}],
    ids=["accion_fichar", "confianza_mayor_que_uno", "confianza_negativa"],
)
def test_un_insight_no_valido_se_descarta_y_los_demas_de_la_respuesta_se_conservan(
    tmp_path, cambio
):
    respuesta = completa(INSIGHT_RAFINA, insight("Pedri", **cambio), INSIGHT_LAMINE)
    openai_falso = OpenAIFalso([("TEXTO_A", respuesta)])

    resultado = extraer([texto("TEXTO_A")], tmp_path, openai_falso)

    assert jugadores(resultado) == ["Rafiña", "la mine"]


def test_las_confianzas_cero_y_uno_son_validas(tmp_path):
    respuesta = completa(insight("Pedri", confianza=0), insight("Gavi", confianza=1))
    openai_falso = OpenAIFalso([("TEXTO_A", respuesta)])

    resultado = extraer([texto("TEXTO_A")], tmp_path, openai_falso)

    assert jugadores(resultado) == ["Gavi", "Pedri"]


def test_las_peticiones_usan_el_modelo_de_extraccion_de_la_configuracion(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FANTASY_MODELO_EXTRACCION", "modelo-x")
    openai_falso = OpenAIFalso([("TEXTO_A", completa(INSIGHT_RAFINA))])

    extraer(
        [texto("TEXTO_A")],
        tmp_path,
        openai_falso,
        modelo=cargar_configuracion().modelo_extraccion,
    )

    assert [peticion["modelo"] for peticion in openai_falso.peticiones] == ["modelo-x"]


def test_de_una_respuesta_cortada_no_se_usa_el_contenido_y_se_devuelven_los_insights_de_las_mitades(
    tmp_path,
):
    contenido_cortado = json.dumps({"insights": [insight("Fantasma")]})
    openai_falso = OpenAIFalso(
        [
            (TEXTO_ENTERO, cortada(contenido_cortado)),
            (PRIMERA_MITAD, completa(INSIGHT_RAFINA)),
            (SEGUNDA_MITAD, completa(INSIGHT_LAMINE)),
        ]
    )

    resultado = extraer([texto("TEXTO_A", contenido=TEXTO_LARGO)], tmp_path, openai_falso)

    assert jugadores(resultado) == ["Rafiña", "la mine"]


def test_las_dos_mitades_de_un_texto_cortado_lo_reparten_sin_partir_ni_perder_palabras(tmp_path):
    openai_falso = OpenAIFalso(
        [
            (TEXTO_ENTERO, cortada()),
            (PRIMERA_MITAD, completa()),
            (SEGUNDA_MITAD, completa()),
        ]
    )

    extraer([texto("TEXTO_A", contenido=TEXTO_LARGO)], tmp_path, openai_falso)

    _, una, otra = (palabras_enteras_en(p["enviado"]) for p in openai_falso.peticiones)
    assert sorted(set(una) | set(otra)) == PALABRAS
    assert 80 <= len(una) <= 120
    assert 80 <= len(otra) <= 120


def test_si_una_mitad_vuelve_a_llegar_cortada_el_texto_queda_fallido_y_no_aporta_insights(
    tmp_path,
):
    openai_falso = OpenAIFalso(
        [
            (TEXTO_ENTERO, cortada()),
            (PRIMERA_MITAD, completa(INSIGHT_RAFINA)),
            (SEGUNDA_MITAD, cortada()),
        ]
    )

    resultado = extraer([texto("TEXTO_A", contenido=TEXTO_LARGO)], tmp_path, openai_falso)

    assert resultado == {
        "insights": [],
        "procesados": [],
        "fallidos": [{"video_id": VIDEO_ID, "chunk": None, "motivo": "respuesta_cortada"}],
    }


def test_si_la_peticion_falla_dos_veces_y_acierta_a_la_tercera_se_devuelven_sus_insights(tmp_path):
    openai_falso = OpenAIFalso([("TEXTO_A", [error_api(), error_api(), completa(INSIGHT_RAFINA)])])

    resultado = extraer([texto("TEXTO_A")], tmp_path, openai_falso)

    assert jugadores(resultado) == ["Rafiña"]


def test_si_falla_el_tercer_intento_el_texto_queda_fallido_por_error_de_api_sin_mas_peticiones(
    tmp_path,
):
    openai_falso = OpenAIFalso([("TEXTO_A", error_api())])

    resultado = extraer([texto("TEXTO_A")], tmp_path, openai_falso)

    assert resultado["fallidos"] == [{"video_id": VIDEO_ID, "chunk": None, "motivo": "error_api"}]
    assert len(openai_falso.peticiones) == 3


@pytest.mark.parametrize(
    "no_valida", [rechazada, a_medias], ids=["contenido_nulo", "json_a_medias"]
)
def test_si_el_contenido_no_es_el_json_esperado_en_los_tres_intentos_el_texto_queda_fallido(
    tmp_path, no_valida
):
    openai_falso = OpenAIFalso([("TEXTO_A", no_valida())])

    resultado = extraer([texto("TEXTO_A")], tmp_path, openai_falso)

    assert resultado == {
        "insights": [],
        "procesados": [],
        "fallidos": [{"video_id": VIDEO_ID, "chunk": None, "motivo": "respuesta_no_valida"}],
    }
    assert len(openai_falso.peticiones) == 3


@pytest.mark.parametrize(
    ("intentos", "motivo"),
    [
        ([error_api, error_api, a_medias], "respuesta_no_valida"),
        ([a_medias, rechazada, error_api], "error_api"),
    ],
    ids=["tercero_no_valido", "tercero_con_error"],
)
def test_los_errores_y_las_respuestas_no_validas_suman_intentos_y_el_motivo_es_el_del_tercero(
    tmp_path, intentos, motivo
):
    respuestas = [*(intento() for intento in intentos), completa(INSIGHT_RAFINA)]
    openai_falso = OpenAIFalso([("TEXTO_A", respuestas)])

    resultado = extraer([texto("TEXTO_A")], tmp_path, openai_falso)

    assert resultado["fallidos"] == [{"video_id": VIDEO_ID, "chunk": None, "motivo": motivo}]
    assert len(openai_falso.peticiones) == 3


def test_un_texto_fallido_no_impide_procesar_los_siguientes(tmp_path):
    openai_falso = OpenAIFalso([("TEXTO_A", error_api()), ("TEXTO_B", completa(INSIGHT_RAFINA))])

    resultado = extraer(
        [texto("TEXTO_A", chunk=1), texto("TEXTO_B", chunk=2)], tmp_path, openai_falso
    )

    assert resultado["insights"] == [{**INSIGHT_RAFINA, "fuente": fuente(chunk=2)}]


def test_el_resultado_se_guarda_en_el_fichero_de_la_jornada_con_su_formato(tmp_path):
    openai_falso = OpenAIFalso([("TEXTO_A", completa(INSIGHT_RAFINA)), ("TEXTO_B", error_api())])
    textos = [texto("TEXTO_A", chunk=1, jornada=3), texto("TEXTO_B", chunk=2, jornada=3)]

    extraer(textos, tmp_path / "datos", openai_falso, jornada=3)

    ruta = tmp_path / "datos" / "2026" / "j03" / "insights.json"
    assert json.loads(ruta.read_text(encoding="utf-8")) == {
        "insights": [{**INSIGHT_RAFINA, "fuente": fuente(chunk=1, jornada=3)}],
        "procesados": [{"video_id": VIDEO_ID, "chunk": 1}],
        "fallidos": [{"video_id": VIDEO_ID, "chunk": 2, "motivo": "error_api"}],
    }


def test_en_una_segunda_ejecucion_solo_se_envia_al_modelo_lo_que_no_estaba_procesado(tmp_path):
    guardar(
        tmp_path,
        {
            "insights": [{**INSIGHT_RAFINA, "fuente": fuente(chunk=1)}],
            "procesados": [
                {"video_id": VIDEO_ID, "chunk": 1},
                {"video_id": "otro_video1", "chunk": None},
            ],
            "fallidos": [],
        },
    )
    openai_falso = OpenAIFalso(
        [
            ("TEXTO_A", completa(insight("Repetido"))),
            ("TEXTO_VACIO", completa(insight("Repetido"))),
            ("TEXTO_B", completa()),
        ]
    )
    textos = [
        texto("TEXTO_A", chunk=1),
        texto("TEXTO_B", chunk=2),
        texto("TEXTO_VACIO", video_id="otro_video1"),
    ]

    extraer(textos, tmp_path, openai_falso)

    [peticion] = openai_falso.peticiones
    assert "TEXTO_B" in peticion["enviado"]


def test_una_segunda_ejecucion_conserva_los_insights_ya_guardados(tmp_path):
    guardado = {
        "insights": [{**INSIGHT_RAFINA, "fuente": fuente(chunk=1)}],
        "procesados": [{"video_id": VIDEO_ID, "chunk": 1}],
        "fallidos": [],
    }
    guardar(tmp_path, guardado)
    openai_falso = OpenAIFalso([("TEXTO_B", completa(INSIGHT_LAMINE))])

    resultado = extraer(
        [texto("TEXTO_A", chunk=1), texto("TEXTO_B", chunk=2)], tmp_path, openai_falso
    )

    assert resultado["insights"] == [
        {**INSIGHT_RAFINA, "fuente": fuente(chunk=1)},
        {**INSIGHT_LAMINE, "fuente": fuente(chunk=2)},
    ]


def test_un_texto_fallido_se_vuelve_a_intentar_y_si_acierta_deja_de_estar_fallido(tmp_path):
    guardar(
        tmp_path,
        {
            "insights": [],
            "procesados": [],
            "fallidos": [{"video_id": VIDEO_ID, "chunk": 2, "motivo": "error_api"}],
        },
    )
    openai_falso = OpenAIFalso([("TEXTO_B", completa(INSIGHT_LAMINE))])

    resultado = extraer([texto("TEXTO_B", chunk=2)], tmp_path, openai_falso)

    assert resultado == {
        "insights": [{**INSIGHT_LAMINE, "fuente": fuente(chunk=2)}],
        "procesados": [{"video_id": VIDEO_ID, "chunk": 2}],
        "fallidos": [],
    }


NO_ES_JSON = '{"insights": ['
SIN_PROCESADOS_NI_FALLIDOS = '{"insights": []}'
LISTA_SUELTA = "[]"

DANADOS = pytest.mark.parametrize(
    "contenido",
    [NO_ES_JSON, SIN_PROCESADOS_NI_FALLIDOS, LISTA_SUELTA],
    ids=["no_es_json", "sin_procesados_ni_fallidos", "lista_suelta"],
)


def guardar_crudo(datos_dir, contenido, jornada=12):
    destino = fichero(datos_dir, jornada)
    destino.parent.mkdir(parents=True)
    destino.write_text(contenido, encoding="utf-8")
    return destino


def test_un_fichero_guardado_que_no_es_json_detiene_la_extraccion_con_un_error_que_dice_su_ruta(
    tmp_path,
):
    ruta = guardar_crudo(tmp_path, NO_ES_JSON)
    openai_falso = OpenAIFalso([("TEXTO_A", completa())])

    with pytest.raises(Exception) as error:
        extraer([texto("TEXTO_A")], tmp_path, openai_falso)

    assert str(ruta) in str(error.value)


@pytest.mark.parametrize(
    "contenido",
    [SIN_PROCESADOS_NI_FALLIDOS, LISTA_SUELTA],
    ids=["sin_procesados_ni_fallidos", "lista_suelta"],
)
def test_un_fichero_guardado_sin_el_formato_detiene_la_extraccion_con_un_error_que_dice_su_ruta(
    tmp_path, contenido
):
    ruta = guardar_crudo(tmp_path, contenido)
    openai_falso = OpenAIFalso([("TEXTO_A", completa())])

    with pytest.raises(Exception) as error:
        extraer([texto("TEXTO_A")], tmp_path, openai_falso)

    assert str(ruta) in str(error.value)


@DANADOS
def test_con_un_fichero_guardado_danado_no_se_hace_ninguna_peticion_al_modelo(tmp_path, contenido):
    guardar_crudo(tmp_path, contenido)
    openai_falso = OpenAIFalso([("TEXTO_A", completa(INSIGHT_RAFINA))])

    with pytest.raises(Exception):  # noqa: B017
        extraer([texto("TEXTO_A")], tmp_path, openai_falso)

    assert openai_falso.peticiones == []


@DANADOS
def test_un_fichero_guardado_danado_queda_exactamente_igual_que_estaba(tmp_path, contenido):
    ruta = guardar_crudo(tmp_path, contenido)
    openai_falso = OpenAIFalso([("TEXTO_A", completa(INSIGHT_RAFINA))])

    with pytest.raises(Exception):  # noqa: B017
        extraer([texto("TEXTO_A")], tmp_path, openai_falso)

    assert ruta.read_bytes() == contenido.encode("utf-8")


def test_al_terminar_en_la_carpeta_de_la_jornada_solo_queda_insights_json(tmp_path):
    openai_falso = OpenAIFalso([("TEXTO_A", completa(INSIGHT_RAFINA)), ("TEXTO_B", error_api())])

    extraer([texto("TEXTO_A", chunk=1), texto("TEXTO_B", chunk=2)], tmp_path, openai_falso)

    carpeta = fichero(tmp_path).parent
    assert [entrada.name for entrada in carpeta.iterdir()] == ["insights.json"]
