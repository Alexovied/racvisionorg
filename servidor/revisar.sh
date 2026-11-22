#!/usr/bin/env bash
# Revisa que un contenedor (rac-app o rac-dev) arrancó bien:  bash revisar.sh rac-dev
set -uo pipefail
C=${1:-rac-dev}
echo "→ Esperando a que $C arranque…"
for i in $(seq 1 30); do
  codigo=$(docker exec "$C" python -c "import urllib.request as u;print(u.urlopen('http://localhost:5000/admin/login',timeout=3).status)" 2>/dev/null) && break
  sleep 3
done
docker ps --filter "name=^$C$" --format '{{.Names}}  {{.Status}}'
if [ "${codigo:-}" = "200" ]; then
  echo "OK: $C responde (login 200)."
else
  echo "ATENCIÓN: $C no responde. Últimas líneas del log:"
  docker logs "$C" --tail 30
  exit 1
fi
if docker logs "$C" --since 5m 2>&1 | grep -q Traceback; then
  echo "ATENCIÓN: hay errores (Traceback) en el log de $C:"
  docker logs "$C" --since 5m 2>&1 | grep -A8 Traceback | tail -30
fi
