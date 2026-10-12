import os
from pathlib import Path

import pytest

from fantasy_copilot.configuracion import ErrorConfiguracion, cargar_configuracion

VARIABLES = (
    "OPENAI_API_KEY",
    "YOUTUBE_API_KEY",
    "WEBSHARE_PROXY_USERNAME",
    "WEBSHARE_PROXY_PASSWORD",
    "FANTASY_DATOS_DIR",
    "FANTASY_MODELO_EXTRACCION",
    "FANTASY_MODELO_NORMALIZACION",
    "FANTASY_MODELO_AGENTE",
    "FANTASY_MODELO_EMBEDDINGS",
)


@pytest.fixture(autouse=True)
def entorno_aislado(tmp_path, monkeypatch):
    """Sin variables del proyecto y en un directorio vacío, sin `.env`."""
    original = dict(os.environ)
    for variable in VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.chdir(tmp_path)
    yield
    os.environ.clear()
    os.environ.update(original)


def test_sin_variables_usa_los_valores_por_defecto():
    config = cargar_configuracion()

    assert config.modelo_extraccion == "gpt-5.4-nano"
    assert config.modelo_normalizacion == "gpt-5.4-nano"
    assert config.modelo_agente == "gpt-5.4-mini"
    assert config.modelo_embeddings == "text-embedding-3-small"
    assert config.datos_dir == Path("datos")


def test_una_variable_cambia_su_valor_y_deja_el_resto_por_defecto(monkeypatch):
    monkeypatch.setenv("FANTASY_MODELO_AGENTE", "otro-modelo")
    monkeypatch.setenv("FANTASY_DATOS_DIR", "/tmp/fc")

    config = cargar_configuracion()

    assert config.modelo_agente == "otro-modelo"
    assert config.datos_dir == Path("/tmp/fc")
    assert config.modelo_extraccion == "gpt-5.4-nano"
    assert config.modelo_normalizacion == "gpt-5.4-nano"
    assert config.modelo_embeddings == "text-embedding-3-small"


def test_devuelve_las_claves_definidas_en_el_entorno(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-prueba")
    monkeypatch.setenv("YOUTUBE_API_KEY", "yt-prueba")

    config = cargar_configuracion()

    assert config.clave_openai == "sk-prueba"
    assert config.clave_youtube == "yt-prueba"


def test_pedir_una_clave_no_definida_da_error_con_el_nombre_de_la_variable():
    config = cargar_configuracion()

    with pytest.raises(ErrorConfiguracion, match="YOUTUBE_API_KEY"):
        config.clave_youtube


def test_pedir_una_clave_vacia_da_error_con_el_nombre_de_la_variable(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    config = cargar_configuracion()

    with pytest.raises(ErrorConfiguracion, match="OPENAI_API_KEY"):
        config.clave_openai


def test_pedir_una_clave_con_solo_espacios_da_error_con_el_nombre_de_la_variable(monkeypatch):
    monkeypatch.setenv("YOUTUBE_API_KEY", "   ")
    config = cargar_configuracion()

    with pytest.raises(ErrorConfiguracion, match="YOUTUBE_API_KEY"):
        config.clave_youtube


def test_sin_variables_de_proxy_no_hay_proxy():
    config = cargar_configuracion()

    assert config.proxy is None


def test_con_usuario_y_contrasena_devuelve_el_proxy(monkeypatch):
    monkeypatch.setenv("WEBSHARE_PROXY_USERNAME", "usuario-prueba")
    monkeypatch.setenv("WEBSHARE_PROXY_PASSWORD", "contrasena-prueba")

    proxy = cargar_configuracion().proxy

    assert (proxy.usuario, proxy.contrasena) == ("usuario-prueba", "contrasena-prueba")


@pytest.mark.parametrize(
    ("definida", "falta"),
    [
        ("WEBSHARE_PROXY_USERNAME", "WEBSHARE_PROXY_PASSWORD"),
        ("WEBSHARE_PROXY_PASSWORD", "WEBSHARE_PROXY_USERNAME"),
    ],
)
def test_proxy_a_medias_da_error_con_la_variable_que_falta(monkeypatch, definida, falta):
    monkeypatch.setenv(definida, "valor-prueba")
    config = cargar_configuracion()

    with pytest.raises(ErrorConfiguracion, match=falta):
        config.proxy


def test_lee_las_variables_de_un_env_en_el_directorio_de_trabajo(tmp_path):
    (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-del-fichero\n", encoding="utf-8")

    config = cargar_configuracion()

    assert config.clave_openai == "sk-del-fichero"


def test_el_entorno_tiene_prioridad_sobre_el_env(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("FANTASY_MODELO_AGENTE=del-fichero\n", encoding="utf-8")
    monkeypatch.setenv("FANTASY_MODELO_AGENTE", "del-entorno")

    config = cargar_configuracion()

    assert config.modelo_agente == "del-entorno"


def test_el_texto_de_la_configuracion_no_contiene_secretos(monkeypatch):
    secretos = {
        "OPENAI_API_KEY": "sk-secreto-openai",
        "YOUTUBE_API_KEY": "yt-secreto-youtube",
        "WEBSHARE_PROXY_PASSWORD": "secreto-contrasena-proxy",
    }
    for variable, valor in secretos.items():
        monkeypatch.setenv(variable, valor)
    monkeypatch.setenv("WEBSHARE_PROXY_USERNAME", "usuario-prueba")

    config = cargar_configuracion()
    texto = f"{config!s} {config!r}"

    for valor in secretos.values():
        assert valor not in texto


def test_los_atributos_de_la_configuracion_como_diccionario_no_contienen_secretos(monkeypatch):
    secretos = {
        "OPENAI_API_KEY": "sk-secreto-openai",
        "YOUTUBE_API_KEY": "yt-secreto-youtube",
        "WEBSHARE_PROXY_PASSWORD": "secreto-contrasena-proxy",
    }
    for variable, valor in secretos.items():
        monkeypatch.setenv(variable, valor)
    monkeypatch.setenv("WEBSHARE_PROXY_USERNAME", "usuario-prueba")

    texto = str(vars(cargar_configuracion()))

    for valor in secretos.values():
        assert valor not in texto


def test_el_texto_del_proxy_no_contiene_la_contrasena(monkeypatch):
    monkeypatch.setenv("WEBSHARE_PROXY_USERNAME", "usuario-prueba")
    monkeypatch.setenv("WEBSHARE_PROXY_PASSWORD", "secreto-contrasena-proxy")

    proxy = cargar_configuracion().proxy
    texto = f"{proxy!s} {proxy!r}"

    assert "secreto-contrasena-proxy" not in texto
