#!/usr/bin/env bash
# Pasa a PRODUCCIÓN el código que está probado en DEV. En el VPS:
#   bash ~/server/rac/servidor/pasar_a_produccion.sh
# Antes hace respaldo del código y de la imagen. Los datos y el .env de producción no se tocan.
# Si algo sale mal:  bash ~/server/rac/servidor/volver_atras.sh
set -euo pipefail
RAC=~/server/rac
PROD=$RAC/deploy
DEV=$RAC/dev
T=$(date +%Y%m%d-%H%M)
EXCLUIR=(--exclude=./data --exclude=./.env --exclude=./docker-compose.yml --exclude=./__pycache__ --exclude='./*.bak*')

docker ps --format '{{.Names}}' | grep -qx rac-dev || { echo "rac-dev no está corriendo. Probá los cambios en dev primero."; exit 1; }

echo "Archivos que cambian en producción:"
cambios=$(diff -rq --exclude=data --exclude=.env --exclude=docker-compose.yml --exclude=__pycache__ \
          --exclude='*.bak*' "$DEV" "$PROD" | grep -v "^Only in $PROD" || true)
if [ -z "$cambios" ]; then echo "  (ninguno: producción ya tiene lo mismo que dev)"; exit 0; fi
echo "$cambios" | sed -e "s|Files $DEV/\(.*\) and .* differ|  cambia:  \1|" -e "s|Only in $DEV/*\(.*\): \(.*\)|  nuevo:   \1/\2|" -e "s|nuevo:   /|nuevo:   |"
read -r -p "¿Pasar esto a producción? Escribí si: " r
[ "$r" = "si" ] || { echo "Cancelado, no se tocó nada."; exit 1; }

echo "→ Respaldo: $RAC/deploy-bak-$T.tgz e imagen deploy-rac-app:bak-$T"
sudo tar -C "$RAC" -czf "$RAC/deploy-bak-$T.tgz" --exclude=deploy/data/alertas --exclude=deploy/__pycache__ deploy
docker tag deploy-rac-app "deploy-rac-app:bak-$T"
echo "$T" > "$RAC/ultimo-respaldo.txt"

echo "→ Copiando el código de dev a producción"
tar -C "$DEV" "${EXCLUIR[@]}" -cf - . | sudo tar -C "$PROD" -xf -

echo "→ Reconstruyendo rac-app"
cd "$PROD" && docker compose up -d --build
if bash "$RAC/servidor/revisar.sh" rac-app; then
  echo "Listo: producción actualizada. Respaldo: $T"
else
  echo "Producción no arrancó bien. Para volver atrás: bash $RAC/servidor/volver_atras.sh"
  exit 1
fi
