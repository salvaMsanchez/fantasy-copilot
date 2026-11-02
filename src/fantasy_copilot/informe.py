NO_HAY_CONJUNTO = "No hay conjunto"


def cifras(resultado: dict) -> dict:
    """Las cifras globales de la evaluación, juntando los casos de recomendación evaluados."""
    evaluados = [caso for caso in resultado["casos"] or [] if caso.get("once")]
    jugadores = [j for caso in evaluados for j in caso["recuperacion"]["jugadores"]]
    resumenes = [j for caso in evaluados for j in caso["fidelidad"]["jugadores"]]
    simples = [caso["once"]["eleccion_simple"] for caso in evaluados]
    return {
        "evaluados": evaluados,
        "recuperacion_precision": _media(j["precision"] for j in jugadores),
        "recuperacion_cobertura": _media(j["cobertura"] for j in jugadores),
        "puntos_recomendado": _total(caso["once"]["recomendado"]["puntos"] for caso in evaluados),
        "puntos_mejor": _total(caso["once"]["mejor_once"]["puntos"] for caso in evaluados),
        "puntos_simple": (
            _total(simple["puntos"] for simple in simples) if None not in simples else None
        ),
        "fidelidad_resumenes": _media(j["fidelidad"] for j in resumenes),
        "resumenes": resumenes,
    }


def resumen(resultado: dict, ruta_informe) -> str:
    totales = cifras(resultado)
    nombres, extraccion = resultado["nombres"], resultado["extraccion"]
    lineas = [f"Evaluación del {resultado['fecha']}"]
    lineas.append(
        "Nombres: "
        + (
            f"precisión {_f(nombres['precision'])} · cobertura {_f(nombres['cobertura'])} · "
            f"eliminados {_f(nombres['porcentaje_eliminados'])}"
            if nombres
            else NO_HAY_CONJUNTO.lower()
        )
    )
    if extraccion:
        lineas += [
            f"Extracción: precisión {_f(extraccion['precision'])} · "
            f"cobertura {_f(extraccion['cobertura'])}",
            f"Fidelidad de los insights: {_f(resultado['fidelidad_insights']['media'])}",
        ]
    else:
        lineas += [f"Extracción: {NO_HAY_CONJUNTO.lower()}"]
    if resultado["casos"] is None:
        lineas.append(f"Recomendación: {NO_HAY_CONJUNTO.lower()}")
    else:
        lineas += [
            f"Recuperación: precisión {_f(totales['recuperacion_precision'])} · "
            f"cobertura {_f(totales['recuperacion_cobertura'])}",
            f"Puntos: recomendado {_p(totales['puntos_recomendado'])} · "
            f"mejor once {_p(totales['puntos_mejor'])} · "
            f"elección simple {_p(totales['puntos_simple'])}",
            f"Fidelidad de los resúmenes: {_f(totales['fidelidad_resumenes'])}",
        ]
    lineas.append(f"Informe: {ruta_informe}")
    return "\n".join(lineas)


def redactar(resultado: dict) -> str:
    totales = cifras(resultado)
    modelos = resultado["modelos"]
    partes = [
        f"# Evaluación del {resultado['fecha']}",
        "## Modelos\n\n"
        f"- Extracción: `{modelos['extraccion']}`\n"
        f"- Normalización: `{modelos['normalizacion']}`\n"
        f"- Agente y juez de las fidelidades: `{modelos['agente']}`\n"
        f"- Embeddings: `{modelos['embeddings']}`",
        _seccion_nombres(resultado),
        _seccion_extraccion(resultado),
        _seccion_recomendacion(resultado, totales),
        _seccion_no_respaldadas(resultado),
    ]
    return "\n\n".join(partes) + "\n"


def _seccion_nombres(resultado: dict) -> str:
    titulo = "## Resolución de nombres"
    nombres = resultado["nombres"]
    if nombres is None:
        return f"{titulo}\n\n{_sin_conjunto(resultado, 'nombres')}"
    return (
        f"{titulo}\n\nConjunto: {len(nombres['nombres'])} nombres.\n\n"
        "| Precisión | Cobertura | Eliminados |\n|---|---|---|\n"
        f"| {_f(nombres['precision'])} | {_f(nombres['cobertura'])} "
        f"| {_f(nombres['porcentaje_eliminados'])} |"
    )


def _seccion_extraccion(resultado: dict) -> str:
    titulo = "## Extracción de insights"
    extraccion = resultado["extraccion"]
    if extraccion is None:
        return (
            f"{titulo}\n\n{_sin_conjunto(resultado, 'extraccion')}\n\n"
            "## Fidelidad de los insights\n\nSin conjunto de extracción, no se mide."
        )
    fidelidad = resultado["fidelidad_insights"]
    return (
        f"{titulo}\n\nConjunto: {len(extraccion['textos'])} textos.\n\n"
        "| Precisión | Cobertura | Correctos | Omitidos | Inventados | Textos fallidos |\n"
        "|---|---|---|---|---|---|\n"
        f"| {_f(extraccion['precision'])} | {_f(extraccion['cobertura'])} "
        f"| {extraccion['correctos']} | {extraccion['omitidos']} | {extraccion['inventados']} "
        f"| {extraccion['fallidos']} |\n\n"
        "## Fidelidad de los insights\n\n"
        f"Media: {_f(fidelidad['media'])} en {len(fidelidad['insights'])} insights"
        f"{_errores(fidelidad['insights'])}."
    )


def _seccion_recomendacion(resultado: dict, totales: dict) -> str:
    titulo = "## Recomendación"
    casos = resultado["casos"]
    if casos is None:
        return f"{titulo}\n\n{_sin_conjunto(resultado, 'recomendacion')}"
    lineas = [
        f"{titulo}\n\nConjunto: {len(casos)} casos, {len(totales['evaluados'])} evaluados.",
        "### Recuperación\n\n"
        f"Precisión media: {_f(totales['recuperacion_precision'])} · "
        f"cobertura media: {_f(totales['recuperacion_cobertura'])}.",
        "### Puntos\n\n"
        "| Jornada | Temporada | Once recomendado | Mejor once | Elección simple "
        "| Fidelidad de los resúmenes |\n|---|---|---|---|---|---|\n"
        + "".join(
            f"| {caso['jornada']} | {caso['temporada']} "
            f"| {_p(caso['once']['recomendado']['puntos'])}"
            f"{'' if caso['once']['recomendado']['valido'] else ' (once no válido)'} "
            f"| {_p(caso['once']['mejor_once']['puntos'])} "
            f"| {_p((caso['once']['eleccion_simple'] or {}).get('puntos'))} "
            f"| {_f(caso['fidelidad']['media'])} |\n"
            for caso in totales["evaluados"]
        )
        + f"| Total | | {_p(totales['puntos_recomendado'])} | {_p(totales['puntos_mejor'])} "
        f"| {_p(totales['puntos_simple'])} | {_f(totales['fidelidad_resumenes'])} |",
        "### Fidelidad de los resúmenes\n\n"
        f"Media: {_f(totales['fidelidad_resumenes'])} en {len(totales['resumenes'])} resúmenes, "
        f"{sum(j['estado'] == 'sin_datos' for j in totales['resumenes'])} sin datos"
        f"{_errores(totales['resumenes'])}.",
    ]
    saltados = [caso for caso in casos if not caso.get("once")]
    if saltados:
        lineas.append(
            "### Casos sin evaluar\n\n"
            + "\n".join(
                f"- Jornada {caso['jornada']} de {caso['temporada']}: {caso['aviso']}"
                for caso in saltados
            )
        )
    return "\n\n".join(lineas)


def _seccion_no_respaldadas(resultado: dict) -> str:
    partes = ["## Afirmaciones no respaldadas"]
    if resultado["extraccion"] is None:
        insights = NO_HAY_CONJUNTO + "."
    else:
        insights = _lista_no_respaldadas(
            (f"{i['jugador']} (vídeo `{i['video_id']}`, trozo {i['chunk']})", i)
            for i in resultado["fidelidad_insights"]["insights"]
        )
    partes.append(f"### De los insights\n\n{insights}")
    if resultado["casos"] is None:
        resumenes = NO_HAY_CONJUNTO + "."
    else:
        resumenes = _lista_no_respaldadas(
            (f"{j['jugador']} (jornada {caso['jornada']} de {caso['temporada']})", j)
            for caso in resultado["casos"]
            if caso.get("once")
            for j in caso["fidelidad"]["jugadores"]
        )
    partes.append(f"### De los resúmenes\n\n{resumenes}")
    return "\n\n".join(partes)


def _lista_no_respaldadas(juzgados) -> str:
    lineas = [
        f"- **{quien}**: «{afirmacion['afirmacion']}». Motivo: {afirmacion['motivo']}"
        for quien, juzgado in juzgados
        for afirmacion in juzgado["no_respaldadas"]
    ]
    return "\n".join(lineas) or "Ninguna."


def _sin_conjunto(resultado: dict, conjunto: str) -> str:
    return f"{NO_HAY_CONJUNTO}: falta `{resultado['ficheros'][conjunto]}`."


def _errores(juzgados: list[dict]) -> str:
    errores = sum(j["estado"] == "error" for j in juzgados)
    return f", {errores} con error del juez" if errores else ""


def _media(valores) -> float | None:
    presentes = [valor for valor in valores if valor is not None]
    return sum(presentes) / len(presentes) if presentes else None


def _total(valores) -> int | None:
    valores = list(valores)
    return sum(valores) if valores else None


def _f(valor: float | None) -> str:
    return "—" if valor is None else f"{valor:.3f}".replace(".", ",")


def _p(valor: int | None) -> str:
    return "—" if valor is None else str(valor)
