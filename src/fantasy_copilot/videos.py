import re

ULTIMA_JORNADA = 38

CANALES = {
    "carrasco": "UCu7ZOzCUgWpvvJzYAdnw0_Q",
    "jornada_perfecta": "UCvMw41zGXyYYPDvyXcHZfug",
}

_PATRON_JORNADA = re.compile(r"\b(?:jornada\s*|j)(\d+)", re.IGNORECASE)
# Una jornada precedida de estas palabras ya se jugó: el vídeo la comenta, no la prepara.
_PATRON_JORNADA_JUGADA = re.compile(
    r"\b(?:tras|despu[eé]s\s+de|lloros\s+de|post(?:\s+fantasy)?)(?:\s+(?:la|el))?\s*$",
    re.IGNORECASE,
)


def extraer_numero_jornada(titulo: str) -> int | None:
    for coincidencia in _PATRON_JORNADA.finditer(titulo):
        jornada = int(coincidencia.group(1))
        jugada = _PATRON_JORNADA_JUGADA.search(titulo[: coincidencia.start()])
        if 1 <= jornada <= ULTIMA_JORNADA and not jugada:
            return jornada
    return None


def es_video_relevante(titulo: str) -> bool:
    if "mundial" in titulo.lower():
        return False
    return extraer_numero_jornada(titulo) is not None


class ErrorYouTube(Exception):
    pass


def obtener_videos(jornada: int, temporada: int, *, cliente) -> list[dict]:
    inicio = f"{temporada}-07-01"
    fin = f"{temporada + 1}-06-30"
    publicados = []
    for canal, canal_id in CANALES.items():
        try:
            publicados += _videos_canal(cliente, canal, canal_id, jornada, temporada, inicio, fin)
        except Exception as error:
            raise ErrorYouTube(
                f"No se pudieron obtener los vídeos del canal {canal}: {error}"
            ) from error
    publicados.sort(key=lambda par: par[0], reverse=True)
    return [video for _, video in publicados]


def _videos_canal(
    cliente, canal: str, canal_id: str, jornada: int, temporada: int, inicio: str, fin: str
) -> list[tuple[str, dict]]:
    respuesta = cliente.channels().list(id=canal_id, part="contentDetails").execute()
    lista_id = respuesta["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]

    publicados = []
    pagina = None
    while True:
        respuesta = (
            cliente.playlistItems()
            .list(playlistId=lista_id, part="snippet", maxResults=50, pageToken=pagina)
            .execute()
        )
        elementos = [elemento["snippet"] for elemento in respuesta.get("items", [])]
        for elemento in elementos:
            titulo = elemento["title"]
            fecha = elemento["publishedAt"][:10]
            if not inicio <= fecha <= fin:
                continue
            if not es_video_relevante(titulo) or extraer_numero_jornada(titulo) != jornada:
                continue
            video_id = elemento["resourceId"]["videoId"]
            video = {
                "canal": canal,
                "video_id": video_id,
                "titulo": titulo,
                "fecha": fecha,
                "jornada": jornada,
                "temporada": temporada,
                "url": f"https://youtube.com/watch?v={video_id}",
            }
            publicados.append((elemento["publishedAt"], video))

        pagina = respuesta.get("nextPageToken")
        anterior_a_la_temporada = elementos and all(
            elemento["publishedAt"][:10] < inicio for elemento in elementos
        )
        if not pagina or anterior_a_la_temporada:
            return publicados
