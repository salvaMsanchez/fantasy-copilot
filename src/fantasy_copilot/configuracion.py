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
    _valores: Mapping[str, str] = field(repr=False)

    @property
    def clave_openai(self) -> str:
        return self._obligatoria("OPENAI_API_KEY")

    @property
    def clave_youtube(self) -> str:
        return self._obligatoria("YOUTUBE_API_KEY")

    @property
    def proxy(self) -> Proxy | None:
        usuario = self._valores.get("WEBSHARE_PROXY_USERNAME")
        contrasena = self._valores.get("WEBSHARE_PROXY_PASSWORD")
        if not usuario and not contrasena:
            return None
        return Proxy(
            usuario=self._obligatoria("WEBSHARE_PROXY_USERNAME"),
            contrasena=self._obligatoria("WEBSHARE_PROXY_PASSWORD"),
        )

    def _obligatoria(self, variable: str) -> str:
        valor = self._valores.get(variable)
        if not valor:
            raise ErrorConfiguracion(f"Falta la variable de entorno {variable}")
        return valor


def cargar_configuracion() -> Configuracion:
    valores = {**dotenv_values(Path.cwd() / ".env"), **os.environ}
    return Configuracion(
        datos_dir=Path(valores.get("FANTASY_DATOS_DIR") or "datos"),
        modelo_extraccion=valores.get("FANTASY_MODELO_EXTRACCION") or "gpt-5.4-nano",
        modelo_normalizacion=valores.get("FANTASY_MODELO_NORMALIZACION") or "gpt-5.4-nano",
        modelo_agente=valores.get("FANTASY_MODELO_AGENTE") or "gpt-5.4-mini",
        modelo_embeddings=valores.get("FANTASY_MODELO_EMBEDDINGS") or "text-embedding-3-small",
        _valores={variable: valores.get(variable) or "" for variable in _SECRETAS},
    )
