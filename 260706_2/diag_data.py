# -*- coding: utf-8 -*-
r"""diag_data.py  --  diagnostico de rutas de datos de RA-TCGF.

Ejecutar desde la MISMA carpeta y del MISMO modo en que lanzas main:
    python diag_data.py
o, si le pasas un data-dir concreto:
    python diag_data.py --data-dir "C:\ruta\a\tus\datos"

No entrena nada. Solo resuelve las rutas EXACTAMENTE igual que main.py y te
dice: (1) cwd, (2) ruta absoluta y existencia de cada fichero, (3) contenido
real del data-dir, (4) donde estan de verdad tus .pkl/.npz/.csv (busqueda
hacia arriba).
"""
import argparse
import os
import sys

# --- valores por defecto IDENTICOS a los de main.py (argparse) -------------- #
DEF_DATA_DIR = "../pems_spatial_kmeans_topK"
DEF_FILES = {
    "flow":   "id_standardize_Env_A_timeseries.csv",
    "coords": "id_node_positions.csv",
    "causal": "subgraph_threshold_0.5.pkl",
    "intent": "intent_graphs_data_20260703_192855.npz",
}


def _ab(p):
    return os.path.abspath(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEF_DATA_DIR)
    ap.add_argument("--flow-csv", default=DEF_FILES["flow"])
    ap.add_argument("--coord-csv", default=DEF_FILES["coords"])
    ap.add_argument("--causal-pkl", default=DEF_FILES["causal"])
    ap.add_argument("--intent-npz", default=DEF_FILES["intent"])
    args = ap.parse_args()

    data_dir = args.data_dir.strip()
    files = {
        "flow":   args.flow_csv.strip(),
        "coords": args.coord_csv.strip(),
        "causal": args.causal_pkl.strip(),
        "intent": args.intent_npz.strip(),
    }

    print("=" * 70)
    print("1) CONTEXTO DE EJECUCION")
    print(f"   cwd (os.getcwd())     : {os.getcwd()}")
    print(f"   este script (__file__): {_ab(__file__)}")
    print(f"   data_dir (config)     : {data_dir!r}")
    print(f"   data_dir -> absoluto  : {_ab(data_dir)}")
    print(f"   data_dir existe?      : {os.path.isdir(_ab(data_dir))}")

    print("=" * 70)
    print("2) RESOLUCION DE CADA FICHERO (join data_dir + nombre)")
    resolved = {}
    all_ok = True
    for key, name in files.items():
        p = os.path.join(data_dir, name)
        pa = _ab(p)
        ok = os.path.exists(pa)
        all_ok = all_ok and ok
        resolved[key] = pa
        print(f"   - {key:7s}: {'OK   ' if ok else 'FALTA'}  {pa}")
    print(f"   => todos presentes? {all_ok}  "
          f"({'usaria datos REALES' if all_ok else 'caeria en SINTETICO'})")

    print("=" * 70)
    print("3) CONTENIDO REAL DEL data_dir")
    dd = _ab(data_dir)
    if os.path.isdir(dd):
        entries = sorted(os.listdir(dd))
        if not entries:
            print("   (vacio)")
        for e in entries:
            full = os.path.join(dd, e)
            tag = "<dir>" if os.path.isdir(full) else f"{os.path.getsize(full):>10d} B"
            print(f"   {tag}  {e}")
    else:
        print(f"   NO EXISTE la carpeta: {dd}")
        print("   -> por eso todo cae en sintetico.")

    print("=" * 70)
    print("4) BUSQUEDA de tus ficheros reales (subiendo hasta 3 niveles)")
    # raiz de busqueda: 3 niveles por encima del cwd
    root = os.getcwd()
    for _ in range(3):
        root = os.path.dirname(root)
    print(f"   raiz de busqueda: {root}")
    wanted_names = set(files.values())
    wanted_ext = (".pkl", ".npz")
    hits_named = []
    hits_ext = []
    n_scanned = 0
    for dirpath, dirnames, filenames in os.walk(root):
        # evita meterte en outputs/ o en .git
        if os.sep + "outputs" in dirpath or os.sep + ".git" in dirpath:
            continue
        for fn in filenames:
            n_scanned += 1
            if fn in wanted_names:
                hits_named.append(os.path.join(dirpath, fn))
            elif fn.lower().endswith(wanted_ext):
                hits_ext.append(os.path.join(dirpath, fn))
    print(f"   ficheros escaneados: {n_scanned}")
    print("   -- coincidencias EXACTAS con los nombres esperados --")
    if hits_named:
        for h in hits_named:
            print(f"      * {h}")
    else:
        print("      (ninguna: los nombres reales NO coinciden con los esperados)")
    print("   -- otros .pkl / .npz encontrados (posibles causal/intent) --")
    for h in hits_ext[:40]:
        print(f"      . {h}")
    if len(hits_ext) > 40:
        print(f"      ... (+{len(hits_ext) - 40} mas)")

    print("=" * 70)
    print("SUGERENCIA:")
    if all_ok:
        print("  Las rutas resuelven OK. Si aun asi sale sintetico, revisa que")
        print("  estas ejecutando ESTE main.py (no otro) y que no hay un fallo")
        print("  de lectura dentro de load_* (prueba: python main.py --strict-data).")
    else:
        if hits_named or hits_ext:
            cand = os.path.dirname((hits_named or hits_ext)[0])
            print("  Tus ficheros parecen estar en otra carpeta. Prueba:")
            print(f'     python main.py --strict-data --data-dir "{cand}" \\')
            print("        --causal-pkl <nombre_real.pkl> --intent-npz <nombre_real.npz> \\")
            print("        --flow-csv <nombre_real.csv> --coord-csv <nombre_real.csv>")
        else:
            print("  No se encontraron .pkl/.npz cerca. Copia tus 4 ficheros a:")
            print(f"     {dd}")
            print("  con EXACTAMENTE estos nombres (o pasa --*-csv/--*-pkl/--*-npz):")
            for k, v in files.items():
                print(f"     {k:7s}: {v}")


if __name__ == "__main__":
    main()
