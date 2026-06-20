"""
modelo.py — CVRP Florida Bebidas · Guanacaste
==============================================
Algoritmo:
  1. Clarke-Wright Savings + 2-opt  →  genera viajes base
  2. Consolidación post-hoc         →  fusiona viajes con espacio libre
     si los cantones de ambos se pueden servir en un solo recorrido
  3. Detalle de paradas             →  cada viaje incluye cuántos pallets
     se descargan en cada cantón

Filosofía de carga máxima
─────────────────────────
• Un viaje que no llega a 24 pallets busca "pasajeros" en otros viajes
  con los que comparta ruta geográfica (ahorro positivo).
• Criterio de consolidación: el viaje combinado debe caber en la
  capacidad Y la distancia combinada no debe superar
  (dist_a + dist_b) × (1 + tolerancia_detour).
• Iteramos la consolidación hasta que no haya más fusiones posibles.
"""

import itertools
import math
import numpy as np
from copy import deepcopy

# ─── DATOS BASE ───────────────────────────────────────────────────────────────

CAPACIDAD_CAMION = 24  # pallets

CANTONES = [
    {"id": 0,  "name": "CD Liberia (Depósito)", "demand": 0,  "lat": 10.6333, "lon": -85.4333},
    {"id": 1,  "name": "Liberia",               "demand": 60, "lat": 10.6333, "lon": -85.4393},
    {"id": 2,  "name": "Nicoya",                "demand": 49, "lat": 10.1500, "lon": -85.4500},
    {"id": 3,  "name": "Santa Cruz",            "demand": 55, "lat": 10.2667, "lon": -85.5833},
    {"id": 4,  "name": "Bagaces",               "demand": 17, "lat": 10.5167, "lon": -85.2500},
    {"id": 5,  "name": "Carrillo",              "demand": 36, "lat": 10.4333, "lon": -85.7000},
    {"id": 6,  "name": "Cañas",                 "demand": 24, "lat": 10.4000, "lon": -85.1167},
    {"id": 7,  "name": "Abangares",             "demand": 16, "lat": 10.2000, "lon": -84.9833},
    {"id": 8,  "name": "Tilarán",               "demand": 16, "lat": 10.4667, "lon": -84.9667},
    {"id": 9,  "name": "Nandayure",             "demand": 8,  "lat": 9.9833,  "lon": -85.2167},
    {"id": 10, "name": "La Cruz",               "demand": 20, "lat": 11.0667, "lon": -85.6167},
    {"id": 11, "name": "Hojancha",              "demand": 7,  "lat": 10.0833, "lon": -85.3000},
]

CANTON_NAME = {c["id"]: c["name"] for c in CANTONES}

# Matriz de distancias por carretera (km) — fuente: Excel base UCR II-1122
DIST_RAW = [
    [  0,  0, 70, 57, 30, 33, 57, 82, 71, 98,  69, 84],
    [  0,  0, 70, 57, 30, 33, 57, 82, 71, 98,  69, 84],
    [ 70, 70,  0, 25, 62, 39, 65, 58, 83, 37, 135, 15],
    [ 57, 57, 25,  0, 60, 25, 74, 76, 93, 62, 116, 39],
    [ 30, 30, 62, 60,  0, 38, 27, 54, 42, 79,  95, 73],
    [ 33, 33, 39, 25, 38,  0, 58, 70, 76, 72,  96, 54],
    [ 57, 57, 65, 74, 27, 58,  0, 31, 19, 68, 120, 71],
    [ 82, 82, 58, 76, 54, 70, 31,  0, 38, 44, 148, 57],
    [ 71, 71, 83, 93, 42, 76, 19, 38,  0, 81, 128, 88],
    [ 98, 98, 37, 62, 79, 72, 68, 44, 81,  0, 166, 26],
    [ 69, 69,135,116, 95, 96,120,148,128,166,   0,150],
    [ 84, 84, 15, 39, 73, 54, 71, 57, 88, 26, 150,  0],
]
DIST = np.array(DIST_RAW, dtype=float)

DEPOSITO = 0

# ─── HELPERS DE RUTA ─────────────────────────────────────────────────────────

def distancia_ruta(ruta, dist=DIST):
    """Distancia total de una ruta [0, c1, ..., cn, 0]."""
    return sum(dist[ruta[i]][ruta[i + 1]] for i in range(len(ruta) - 1))


def dos_opt(ruta, dist=DIST):
    """2-opt: mejora local de la secuencia de visitas."""
    mejor = ruta[:]
    mejoro = True
    while mejoro:
        mejoro = False
        n = len(mejor)
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                nueva = mejor[:i] + mejor[i:j + 1][::-1] + mejor[j + 1:]
                if distancia_ruta(nueva, dist) < distancia_ruta(mejor, dist) - 1e-6:
                    mejor = nueva
                    mejoro = True
    return mejor


def mejor_insercion(ruta_base, nuevo_nodo, dist=DIST):
    """
    Devuelve la ruta con el nuevo_nodo insertado en la posición de menor coste.
    ruta_base incluye depósito al inicio y al final.
    """
    mejor_ruta = None
    mejor_d = float("inf")
    for pos in range(1, len(ruta_base)):
        candidata = ruta_base[:pos] + [nuevo_nodo] + ruta_base[pos:]
        d = distancia_ruta(candidata, dist)
        if d < mejor_d:
            mejor_d = d
            mejor_ruta = candidata
    return mejor_ruta, mejor_d


# ─── FASE 1: CLARKE-WRIGHT CON ENTREGAS DIVIDIDAS ────────────────────────────

def clarke_wright(demandas, dist, capacidad, deposito=DEPOSITO):
    """
    Heurística Clarke-Wright para CVRP con split deliveries.
    Retorna lista de viajes: cada viaje es dict con
      'nodos'   : [deposito, c1, c2, ..., deposito]  (enteros)
      'carga'   : dict {canton_id: pallets_entregados}
      'load'    : total pallets
      'dist'    : km totales
    """
    clientes = [c for c in demandas if c != deposito and demandas[c] > 0]

    # Expandir cantones con demanda > capacidad en sub-clientes
    expandidos = []   # [(canton_id, chunk_pallets), ...]
    for c in clientes:
        d = demandas[c]
        while d > 0:
            chunk = min(d, capacidad)
            expandidos.append((c, chunk))
            d -= chunk

    # Cada sub-cliente arranca como ruta propia
    # Usamos índice único para identificar sub-clientes en las rutas
    n = len(expandidos)
    rutas = [
        {
            "sub_nodes": [i],          # índices en 'expandidos'
            "load": expandidos[i][1],
        }
        for i in range(n)
    ]

    # Ahorros: s(i,j) = d(0,ci) + d(0,cj) - d(ci,cj)
    ahorros = []
    for ia, ib in itertools.combinations(range(n), 2):
        ci, _ = expandidos[ia]
        cj, _ = expandidos[ib]
        s = dist[deposito][ci] + dist[deposito][cj] - dist[ci][cj]
        if s > 0:
            ahorros.append((s, ia, ib))
    ahorros.sort(key=lambda x: -x[0])

    # Mapa sub_node → índice de ruta
    def ruta_de(sub_idx):
        for r in rutas:
            if sub_idx in r["sub_nodes"]:
                return r
        return None

    def es_extremo(ruta, sub_idx):
        sn = ruta["sub_nodes"]
        return sn[0] == sub_idx or sn[-1] == sub_idx

    for s, ia, ib in ahorros:
        ra = ruta_de(ia)
        rb = ruta_de(ib)
        if ra is None or rb is None or ra is rb:
            continue
        if ra["load"] + rb["load"] > capacidad:
            continue
        if not es_extremo(ra, ia) or not es_extremo(rb, ib):
            continue

        sna = ra["sub_nodes"]
        snb = rb["sub_nodes"]
        # Orientar: ia al final de ra, ib al inicio de rb
        if sna[0] == ia:
            sna = sna[::-1]
        if snb[-1] == ib:
            snb = snb[::-1]

        merged = {
            "sub_nodes": sna + snb,
            "load": ra["load"] + rb["load"],
        }
        rutas.remove(ra)
        rutas.remove(rb)
        rutas.append(merged)

    # Construir rutas finales con nodos reales
    viajes = []
    for r in rutas:
        # Convertir sub-clientes → cantones (puede haber repetidos)
        canton_seq = [expandidos[si][0] for si in r["sub_nodes"]]
        # Ruta: depósito → cantones → depósito
        ruta_nodos = [deposito] + canton_seq + [deposito]
        ruta_nodos = dos_opt(ruta_nodos, dist)

        # Carga por cantón en este viaje
        carga = {}
        for si in r["sub_nodes"]:
            cid, chunk = expandidos[si]
            carga[cid] = carga.get(cid, 0) + chunk

        viajes.append({
            "nodos": ruta_nodos,
            "carga": carga,
            "load":  r["load"],
            "dist":  distancia_ruta(ruta_nodos, dist),
        })

    return viajes


# ─── FASE 2: CONSOLIDACIÓN — MAXIMIZAR CARGA ─────────────────────────────────

def consolidar(viajes, capacidad, dist=DIST, deposito=DEPOSITO,
               tolerancia_detour=0.30):
    """
    Post-procesamiento: intenta fusionar pares de viajes incompletos.

    Un viaje A (carga < capacidad) absorbe los cantones del viaje B si:
      1. load_A + load_B <= capacidad
      2. La nueva distancia no supera max(dist_A, dist_B) * (1 + tolerancia)

    La fusión garantiza que todos los pallets se entreguen y el camión
    salga lo más lleno posible, reduciendo el número total de viajes.

    Retorna la lista de viajes consolidada.
    """
    viajes = deepcopy(viajes)
    mejoro = True

    while mejoro:
        mejoro = False
        # Ordenar: viajes más vacíos primero (los que más necesitan absorber)
        viajes.sort(key=lambda v: v["load"])
        n = len(viajes)

        for i in range(n):
            for j in range(i + 1, n):
                va = viajes[i]
                vb = viajes[j]

                carga_combinada = va["load"] + vb["load"]
                if carga_combinada > capacidad:
                    continue

                # Insertar cada cantón de vb en la ruta de va
                nueva_ruta = list(va["nodos"])
                for canton in vb["nodos"][1:-1]:  # sin depósitos
                    nueva_ruta, _ = mejor_insercion(nueva_ruta, canton, dist)

                nueva_ruta = dos_opt(nueva_ruta, dist)
                nueva_dist = distancia_ruta(nueva_ruta, dist)

                # Límite de detour: la ruta fusionada no debe ser
                # mucho más larga que la mayor de las dos originales
                limite = max(va["dist"], vb["dist"]) * (1 + tolerancia_detour)
                # Si ambas rutas son muy cortas, usar suma como límite
                limite = max(limite, va["dist"] + vb["dist"])

                if nueva_dist <= limite + 1e-6:
                    # Fusionar carga
                    carga_nueva = dict(va["carga"])
                    for cid, pallets in vb["carga"].items():
                        carga_nueva[cid] = carga_nueva.get(cid, 0) + pallets

                    viajes[i] = {
                        "nodos": nueva_ruta,
                        "carga": carga_nueva,
                        "load":  carga_combinada,
                        "dist":  nueva_dist,
                    }
                    viajes.pop(j)
                    mejoro = True
                    break  # reiniciar el bucle exterior

            if mejoro:
                break

    return viajes


# ─── FASE 3: DETALLE DE PARADAS ───────────────────────────────────────────────

def detallar_paradas(viaje, capacidad):
    """
    Enriquece un viaje con información de cada parada:
      - pallets a bordo al llegar
      - pallets descargados
      - pallets a bordo al salir
      - % de llenado
    Retorna lista de dicts, una entrada por nodo visitado (incluyendo CD).
    """
    paradas = []
    a_bordo = viaje["load"]  # sale lleno (o lo que lleva)
    carga = viaje["carga"]

    for idx, nodo in enumerate(viaje["nodos"]):
        if nodo == DEPOSITO and idx == 0:
            paradas.append({
                "orden":       1,
                "canton_id":   DEPOSITO,
                "canton":      CANTON_NAME[DEPOSITO],
                "llega_con":   a_bordo,
                "descarga":    0,
                "sale_con":    a_bordo,
                "pct_lleno":   round(a_bordo / capacidad * 100, 1),
                "tipo":        "SALIDA",
            })
        elif nodo == DEPOSITO and idx > 0:
            paradas.append({
                "orden":       idx + 1,
                "canton_id":   DEPOSITO,
                "canton":      CANTON_NAME[DEPOSITO],
                "llega_con":   a_bordo,
                "descarga":    0,
                "sale_con":    0,
                "pct_lleno":   0.0,
                "tipo":        "REGRESO",
            })
        else:
            entrega = carga.get(nodo, 0)
            paradas.append({
                "orden":       idx + 1,
                "canton_id":   nodo,
                "canton":      CANTON_NAME.get(nodo, f"Nodo {nodo}"),
                "llega_con":   a_bordo,
                "descarga":    entrega,
                "sale_con":    a_bordo - entrega,
                "pct_lleno":   round((a_bordo - entrega) / capacidad * 100, 1),
                "tipo":        "ENTREGA",
            })
            a_bordo -= entrega

    return paradas


# ─── FASE 4: ASIGNACIÓN A CAMIONES FÍSICOS (FFD) ─────────────────────────────

def asignar_camiones(viajes, velocidad_kmh=60, recarga_min=20, jornada_h=8,
                     capacidad=CAPACIDAD_CAMION):
    """
    Asigna viajes a camiones físicos (FFD por tiempo).
    Un camión puede hacer varios viajes por jornada si le alcanza el tiempo.
    """
    recarga_h = recarga_min / 60.0
    camiones = []

    # Ordenar viajes: los más largos primero (FFD)
    viajes_ord = sorted(viajes, key=lambda v: -v["dist"])

    for viaje in viajes_ord:
        t_viaje = viaje["dist"] / velocidad_kmh if velocidad_kmh > 0 else 0
        colocado = False

        for cam in camiones:
            t_acum = sum(v["dist"] / velocidad_kmh for v in cam["viajes"])
            t_acum += recarga_h * len(cam["viajes"])
            if t_acum + t_viaje <= jornada_h:
                cam["viajes"].append(viaje)
                colocado = True
                break

        if not colocado:
            camiones.append({"viajes": [viaje]})

    resultado = []
    for i, cam in enumerate(camiones):
        t_cond   = sum(v["dist"] / velocidad_kmh for v in cam["viajes"])
        t_rec    = recarga_h * (len(cam["viajes"]) - 1)
        t_total  = t_cond + t_rec
        resultado.append({
            "camion_id":        f"C-{i + 1:02d}",
            "viajes":           cam["viajes"],
            "n_viajes":         len(cam["viajes"]),
            "tiempo_h":         round(t_total, 2),
            "uso_jornada_pct":  round(t_total / jornada_h * 100, 1),
            "pallets_totales":  sum(v["load"] for v in cam["viajes"]),
            "dist_total_km":    round(sum(v["dist"] for v in cam["viajes"]), 1),
        })

    return resultado


# ─── FUNCIÓN PRINCIPAL ────────────────────────────────────────────────────────

def resolver_cvrp(demandas_custom=None, capacidad=CAPACIDAD_CAMION,
                  velocidad=60, recarga_min=20, jornada_h=8,
                  tolerancia_detour=0.30):
    """
    Pipeline completo:
      1. Clarke-Wright + 2-opt
      2. Consolidación para maximizar carga
      3. Detalle de paradas por viaje
      4. Asignación FFD a camiones físicos
      5. KPIs

    Retorna
    -------
    dict con claves:
      'viajes'   : lista de viajes (con 'paradas' detalladas)
      'camiones' : asignación física
      'kpis'     : métricas globales
    """
    # Construir demandas
    if demandas_custom:
        demandas = {c["id"]: demandas_custom.get(c["id"], c["demand"]) for c in CANTONES}
    else:
        demandas = {c["id"]: c["demand"] for c in CANTONES}
    demandas[DEPOSITO] = 0

    # Fase 1: Clarke-Wright
    viajes = clarke_wright(demandas, DIST, capacidad)

    # Fase 2: Consolidación
    viajes = consolidar(viajes, capacidad, DIST, DEPOSITO, tolerancia_detour)

    # Fase 3: Detallar paradas
    for v in viajes:
        v["paradas"] = detallar_paradas(v, capacidad)

    # Fase 4: Asignación a camiones
    camiones = asignar_camiones(
        viajes, velocidad_kmh=velocidad,
        recarga_min=recarga_min, jornada_h=jornada_h,
        capacidad=capacidad,
    )

    # KPIs
    demanda_total = sum(demandas[i] for i in demandas if i != DEPOSITO)
    dist_total    = sum(v["dist"] for v in viajes)
    util_prom     = (sum(v["load"] for v in viajes) / (len(viajes) * capacidad) * 100
                     if viajes else 0)

    kpis = {
        "demanda_total":         demanda_total,
        "n_viajes":              len(viajes),
        "n_camiones":            len(camiones),
        "dist_total_km":         round(dist_total, 1),
        "dist_media_km":         round(dist_total / len(viajes), 1) if viajes else 0,
        "viajes_llenos":         sum(1 for v in viajes if v["load"] == capacidad),
        "viajes_casi_llenos":    sum(1 for v in viajes if v["load"] / capacidad >= 0.80),
        "utilizacion_prom_pct":  round(util_prom, 1),
        "viajes_min_teoricos":   math.ceil(demanda_total / capacidad),
        "pallets_verificados":   sum(
            sum(v["carga"].values()) for v in viajes
        ),
    }

    return {"viajes": viajes, "camiones": camiones, "kpis": kpis}
