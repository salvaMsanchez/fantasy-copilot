import hashlib
import json
import uuid

import chromadb

TOP_K = 10
TEXTOS_POR_LOTE = 100
CRITERIOS = (
    ("forma_actual", "Forma actual"),
    ("dinamica_equipo", "Dinámica equipo"),
    ("calendario_rival", "Calendario rival"),
    ("prediccion_puntos", "Predicción puntos"),
)


class ErrorInsightNoNormalizado(Exception):
    pass


class Indice:
    def __init__(self, *, cliente, modelo: str) -> None:
        self._cliente = cliente
        self._modelo = modelo
        self._coleccion = chromadb.EphemeralClient().create_collection(
            name=f"insights-{uuid.uuid4().hex}",
            configuration={"hnsw": {"space": "cosine"}},
        )

    def indexar(self, insights: list[dict]) -> None:
        sin_normalizar = [insight for insight in insights if not insight.get("jugador_id")]
        if sin_normalizar:
            raise ErrorInsightNoNormalizado(
                f"{len(sin_normalizar)} insights no tienen jugador_id: "
                "solo se indexan insights normalizados"
            )
        por_id = {_identificador(insight): insight for insight in insights}
        if not por_id:
            return
        textos = [_texto(insight) for insight in por_id.values()]
        self._coleccion.upsert(
            ids=list(por_id),
            documents=textos,
            embeddings=self._embeddings(textos),
            metadatas=[_metadatos(insight) for insight in por_id.values()],
        )

    def consultar(
        self, pregunta: str, *, jornada: int, temporada: int, top_k: int = TOP_K
    ) -> list[dict]:
        resultados = self._coleccion.query(
            query_embeddings=self._embeddings([pregunta]),
            n_results=top_k,
            where={"$and": [{"jornada": jornada}, {"temporada": temporada}]},
        )
        return [
            {"texto": texto, "metadatos": metadatos, "distancia": distancia}
            for texto, metadatos, distancia in zip(
                resultados["documents"][0],
                resultados["metadatas"][0],
                resultados["distances"][0],
                strict=True,
            )
        ]

    def _embeddings(self, textos: list[str]) -> list[list[float]]:
        vectores = []
        for inicio in range(0, len(textos), TEXTOS_POR_LOTE):
            respuesta = self._cliente.embeddings.create(
                model=self._modelo, input=textos[inicio : inicio + TEXTOS_POR_LOTE]
            )
            vectores.extend(dato.embedding for dato in respuesta.data)
        return vectores


def _identificador(insight: dict) -> str:
    contenido = json.dumps(insight, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(contenido.encode("utf-8")).hexdigest()


def _texto(insight: dict) -> str:
    criterios = insight.get("criterios") or {}
    fuente = insight["fuente"]
    partes = [
        f"{insight['jugador']}, {insight.get('posicion', '')}.",
        f"Recomendación: {insight['accion']}.",
        f"Razón: {insight.get('razon_corta', '')}.",
        *(
            f"{etiqueta}: {criterios[clave]}."
            for clave, etiqueta in CRITERIOS
            if criterios.get(clave)
        ),
        f"Fuente: {fuente['canal']}, Jornada {fuente['jornada']}.",
    ]
    return " ".join(partes)


def _metadatos(insight: dict) -> dict:
    fuente = insight["fuente"]
    return {
        "jugador": insight["jugador"],
        "jugador_id": insight["jugador_id"],
        "equipo": insight["equipo"],
        "posicion": insight["posicion"],
        "accion": insight["accion"],
        "confianza": float(insight["confianza"]),
        "canal": fuente["canal"],
        "fecha": fuente["fecha"],
        "jornada": int(fuente["jornada"]),
        "temporada": int(fuente["temporada"]),
        "url": fuente["url"],
    }
