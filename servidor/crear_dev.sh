#!/usr/bin/env bash
# Crea el entorno de DESARROLLO (contenedor rac-dev) al lado de producción (rac-app).
# Se corre UNA sola vez, en el VPS:   bash ~/server/rac/servidor/crear_dev.sh
#
# - Código: copia del de producción (después se actualiza con subir_a_dev.sh desde la PC).
# - Datos: copia de los de producción (usuarios, eventos, wearables), pero en su propia carpeta.
#   Lo que hagas en dev no toca los datos reales.
# - Sin Telegram, MQTT, n8n ni correo: las alertas de prueba no le llegan a nadie.
# - Las cámaras IP/USB quedan desactivadas en dev (si no, producción y dev se conectarían a
#   la misma cámara). Si querés probar una, activala desde el panel de dev.
set -euo pipefail
PROD=~/server/rac/deploy
DEV=~/server/rac/dev

if [ -d "$DEV" ]; then
  echo "Ya existe $DEV. Para rehacerlo desde cero:"
  echo "  cd $DEV && docker compose down && cd .. && sudo rm -rf $DEV && bash $0"
  exit 1
fi
docker network inspect proxy-network >/dev/null

echo "→ Copiando el código de producción a $DEV"
mkdir -p "$DEV"
tar -C "$PROD" --exclude=./data --exclude=./.env --exclude=./docker-compose.yml \
    --exclude='./*.bak*' --exclude=./__pycache__ -cf - . | tar -C "$DEV" -xf -

echo "→ Copiando los datos ($(sudo du -sh "$PROD/data" | cut -f1))"
sudo cp -a "$PROD/data" "$DEV/data"
sudo rm -f "$DEV"/data/reporte-*.lock

echo "→ Apagando las cámaras IP/USB en dev"
if sudo test -f "$DEV/data/cameras.json"; then
  sudo python3 - "$DEV/data/cameras.json" <<'PY'
import json, sys
ruta = sys.argv[1]
camaras = json.load(open(ruta, encoding="utf-8"))
for c in camaras:
    if c.get("type") in ("usb", "ip"):
        c["active"] = False
json.dump(camaras, open(ruta, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
PY
fi

echo "→ Armando el .env de dev (sin Telegram, MQTT, n8n, correo ni reporte automático)"
grep -vE '^\s*(TELEGRAM_|MQTT_|N8N_|SMTP_|SECRET_KEY|REPORTE_HORA|RAC_ENTORNO)' "$PROD/.env" > "$DEV/.env" || true
{
  echo "SECRET_KEY=$(openssl rand -hex 32)"
  echo "RAC_ENTORNO=desarrollo"
} >> "$DEV/.env"
chmod 600 "$DEV/.env"

cat > "$DEV/docker-compose.yml" <<'YML'
# DESARROLLO: mismo Dockerfile que producción, datos y .env aparte.
# Lo publica el proxy (Nginx Proxy Manager) en dev.racvision.org → rac-dev:5000.
name: rac-dev
services:
  rac-dev:
    build: .
    image: rac-dev
    container_name: rac-dev
    restart: unless-stopped
    env_file:
      - .env
    volumes:
      - ./data:/app/data
    # Que dev nunca le saque memoria o CPU a producción
    mem_limit: 2g
    cpus: 1.5
    networks:
      - proxy-network

networks:
  proxy-network:
    external: true
YML

echo "→ Levantando rac-dev (la primera vez tarda unos minutos)"
cd "$DEV" && docker compose up -d --build
bash ~/server/rac/servidor/revisar.sh rac-dev
