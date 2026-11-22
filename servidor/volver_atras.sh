#!/usr/bin/env bash
# Vuelve PRODUCCIÓN al último respaldo (o a uno en particular):
#   bash ~/server/rac/servidor/volver_atras.sh               → el último
#   bash ~/server/rac/servidor/volver_atras.sh 20261122-1930 → ese
# Restaura el código y la imagen. Los datos (usuarios, eventos) quedan como están.
set -euo pipefail
RAC=~/server/rac
T=${1:-$(cat "$RAC/ultimo-respaldo.txt")}
[ -f "$RAC/deploy-bak-$T.tgz" ] || { echo "No existe $RAC/deploy-bak-$T.tgz"; ls "$RAC"/deploy-bak-*.tgz; exit 1; }
docker image inspect "deploy-rac-app:bak-$T" >/dev/null

echo "→ Restaurando el código de $T"
sudo tar -C "$RAC" -xzf "$RAC/deploy-bak-$T.tgz" --exclude=deploy/data
echo "→ Volviendo a la imagen deploy-rac-app:bak-$T"
docker tag "deploy-rac-app:bak-$T" deploy-rac-app
cd "$RAC/deploy" && docker compose up -d --no-build --force-recreate
bash "$RAC/servidor/revisar.sh" rac-app
