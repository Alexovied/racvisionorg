# Producción y desarrollo en el VPS

| | Producción | Desarrollo |
|---|---|---|
| Dirección | https://racvision.org | https://dev.racvision.org |
| Carpeta | `~/server/rac/deploy` | `~/server/rac/dev` |
| Contenedor | `rac-app` | `rac-dev` (máx. 2 GB de RAM y 1,5 CPU) |
| Datos | `deploy/data` | `dev/data` (copia de producción del día que se creó) |
| Telegram, MQTT, n8n, correo | sí | no: las alertas de prueba no le llegan a nadie |
| Cámaras IP/USB | activas | apagadas al crear dev (activá una desde el panel de dev si querés probarla) |
| Panel | normal | franja naranja «DESARROLLO» arriba, y Google no lo indexa |

## Cómo se trabaja

1. **Subir a dev** (Git Bash en la PC): `cd ~/Desktop/rac && bash servidor/subir_a_dev.sh` (sube `main`;
   para otra rama: `bash servidor/subir_a_dev.sh nombre-de-rama`). Producción no se toca.
2. **Probar** en https://dev.racvision.org.
3. **Pasar a producción** (en el VPS): `bash ~/server/rac/servidor/pasar_a_produccion.sh`. Muestra qué archivos
   cambian, pide escribir `si`, hace respaldo del código y de la imagen, copia el código de dev y reinicia `rac-app`.
4. **Si algo sale mal**: `bash ~/server/rac/servidor/volver_atras.sh` (vuelve al último respaldo; los datos no se tocan).

`subir_a_dev.sh` no sube la landing (`templates/landing.html`) ni `requirements.txt`, porque el servidor tiene los
suyos, ni los modelos (~100 MB). Si cambiaste un modelo: `MODELOS=1 bash servidor/subir_a_dev.sh`.

Para ver el estado: `bash ~/server/rac/servidor/revisar.sh rac-dev` (o `rac-app`). Para ver qué tiene dev: `cat ~/server/rac/dev/VERSION`.

## Crear dev (una sola vez)

1. Desde la PC, en Git Bash, copiar estos scripts al VPS y crear dev:
   ```
   cd ~/Desktop/rac && git fetch origin main && git archive FETCH_HEAD servidor | ssh ubuntu@40.160.138.65 "mkdir -p ~/server/rac && tar -C ~/server/rac -xf - && bash ~/server/rac/servidor/crear_dev.sh"
   ```
2. **DNS** (Cloudflare): registro `A`, nombre `dev`, contenido `40.160.138.65`, proxy activado (nube naranja).
3. **Proxy** (Nginx Proxy Manager, el contenedor `proxy-app-1`): su panel está en el puerto 81 solo del servidor.
   Abrir un túnel desde la PC con `ssh -L 8181:127.0.0.1:81 ubuntu@40.160.138.65`, entrar a http://localhost:8181 y
   en **Hosts → Proxy Hosts → Add Proxy Host**: dominio `dev.racvision.org`, esquema `http`, host `rac-dev`,
   puerto `5000`, activar *Websockets Support* y *Block Common Exploits*. En **SSL** pedir un certificado nuevo
   y activar *Force SSL* (lo mismo que tenga el host de `racvision.org`).

Para borrar dev: `cd ~/server/rac/dev && docker compose down && cd .. && sudo rm -rf dev`.
