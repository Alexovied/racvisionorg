#!/usr/bin/env bash
# Sube una rama de GitHub al entorno de DESARROLLO. En Git Bash, en la PC:
#   cd ~/Desktop/rac && bash servidor/subir_a_dev.sh            → sube main
#   cd ~/Desktop/rac && bash servidor/subir_a_dev.sh mi-rama    → sube esa rama
#   MODELOS=1 bash servidor/subir_a_dev.sh                       → también los modelos (~100 MB)
# No toca producción. Después probás en https://dev.racvision.org y, si anda,
# en el VPS:  bash ~/server/rac/servidor/pasar_a_produccion.sh
set -euo pipefail
RAMA=${1:-main}
SERVIDOR=ubuntu@40.160.138.65
cd "$(git rev-parse --show-toplevel)"
git fetch origin "$RAMA"
echo "→ Empaquetando $RAMA ($(git rev-parse --short FETCH_HEAD))"
# Sin conversión a CRLF, y sin lo que el servidor tiene propio (landing editada a mano, requirements)
FUERA=(':!templates/landing.html' ':!requirements.txt' ':!data' ':!tests' ':!docs' ':!colab'
       ':!entrenamiento' ':!firmware')
# Los modelos casi nunca cambian y pesan ~100 MB: dev ya los tiene de la copia inicial
[ "${MODELOS:-0}" = 1 ] || FUERA+=(':!*.onnx' ':!*.keras' ':!*_int8.bin' ':!*_int8.xml')
git -c core.autocrlf=false archive --format=tar.gz -o /tmp/rac-dev.tgz FETCH_HEAD -- . "${FUERA[@]}"
scp /tmp/rac-dev.tgz "$SERVIDOR:/tmp/rac-dev.tgz"
ssh "$SERVIDOR" "tar -C ~/server/rac/dev -xzf /tmp/rac-dev.tgz && rm /tmp/rac-dev.tgz \
  && mkdir -p ~/server/rac/servidor && cp ~/server/rac/dev/servidor/*.sh ~/server/rac/servidor/ \
  && echo $RAMA \$(date '+%F %R') > ~/server/rac/dev/VERSION \
  && cd ~/server/rac/dev && docker compose up -d --build && bash ~/server/rac/servidor/revisar.sh rac-dev"
echo "Listo: probalo en https://dev.racvision.org"
