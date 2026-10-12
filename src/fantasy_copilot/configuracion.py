import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values

_SECRETAS = (
    "OPENAI_API_KEY",
    "YOUTUBE_API_KEY",
    "WEBSHARE_PROXY_USERNAME",
    "WEBSHARE_PROXY_PASSWORD",
)


class ErrorConfiguracion(Exception):
    pass


class _Secretos:
    def __init__(self, valores: Mapping[str, str]) -> None:
        self._valores = dict(valores)

    def get(self, variable: str) -> str:
        return self._valores.get(variable, "")

    def __eq__(self, otro: object) -> bool:
        return isinstance(otro, _Secretos) and self._valores == otro._valores

    def __hash__(self) -> int:
        return hash(tuple(sorted(self._valores.items())))

    def __repr__(self) -> str:
        return "<oculto>"


@dataclass(frozen=True)
class Proxy:
    usuario: str
    contrasena: str = field(repr=False)


@dataclass(frozen=True)
class Configuracion:
    datos_dir: Path
    modelo_extraccion: str
    modelo_normalizacion: str
    modelo_agente: str
    modelo_embeddings: str
    _secretos: _Secretos = field(repr=False)

    @property
    def clave_openai(self) -> str:
        return self._obligatoria("OPENAI_API_KEY")

    @property
    def clave_youtube(self) -> str:
        return self._obligatoria("YOUTUBE_API_KEY")

    @property
    def proxy(self) -> Proxy | None:
        usuario = self._secretos.get("WEBSHARE_PROXY_USERNAME")
        contrasena = self._secretos.get("WEBSHARE_PROXY_PASSWORD")
        if not usuario and not contrasena:
            return None
        return Proxy(
            usuario=self._obligatoria("WEBSHARE_PROXY_USERNAME"),
            contrasena=self._obligatoria("WEBSHARE_PROXY_PASSWORD"),
        )

    def _obligatoria(self, variable: str) -> str:
        valor = self._secretos.get(variable)
        if not valor:
            raise ErrorConfiguracion(f"Falta la variable de entorno {variable}")
        return valor


def cargar_configuracion() -> Configuracion:
    entorno = {**dotenv_values(Path.cwd() / ".env"), **os.environ}

    def valor(variable: str, por_defecto: str = "") -> str:
        return (entorno.get(variable) or "").strip() or por_defecto

    return Configuracion(
        datos_dir=Path(valor("FANTASY_DATOS_DIR", "datos")),
        modelo_extraccion=valor("FANTASY_MODELO_EXTRACCION", "gpt-5.4-nano"),
        modelo_normalizacion=valor("FANTASY_MODELO_NORMALIZACION", "gpt-5.4-nano"),
        modelo_agente=valor("FANTASY_MODELO_AGENTE", "gpt-5.4-mini"),
        modelo_embeddings=valor("FANTASY_MODELO_EMBEDDINGS", "text-embedding-3-small"),
        _secretos=_Secretos({variable: valor(variable) for variable in _SECRETAS}),
    )
