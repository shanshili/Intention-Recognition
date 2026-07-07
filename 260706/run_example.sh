#!/usr/bin/env bash
# Ejemplo de ejecucion del pipeline RA-TCGF.
# Ajusta --data-dir a la carpeta con tus ficheros reales:
#   id_standardize_Env_A_timeseries.csv, id_node_positions.csv,
#   causal_subgraph.pkl, intent_graphs.npz
set -e

python -m ratcgf.main \
    --data-dir ../pems_spatial_kmeans_topK \
    --epochs 40 \
    --device cuda \
    --gnn gat \
    --show-steps 97
