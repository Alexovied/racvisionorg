// Reglas de EPP compartidas por /app y el monitoreo del panel.
// Casco y chaleco se evalúan por separado: cada uno tiene su propia ventana de votos,
// así faltar uno no tapa que el otro está bien, y cualquiera de los dos puede estar apagado.
const EPP = (() => {
    // Ventana de 8 frames: alerta con 60 % de votos "falta"; OK con 70 % y ventana llena.
    const WINDOW_SIZE = 8, THRESH_FALTA = 0.60, THRESH_OK = 0.70;
    const NOMBRES = { casco: 'casco', chaleco: 'chaleco' };

    // Estado de un elemento en una persona: 'ok', 'falta' o null (no se puede juzgar)
    function estadoCasco(c, minConf) {
        if (c.con_casco == null || Math.max(c.con_casco, c.sin_casco) < minConf) return null;
        return c.con_casco > c.sin_casco ? 'ok' : 'falta';
    }
    const estadoChaleco = c => c.chaleco === true ? 'ok' : c.chaleco === false ? 'falta' : null;

    function evaluar(c, minConf) {
        return { casco: estadoCasco(c, minConf), chaleco: estadoChaleco(c) };
    }

    // Texto y color del recuadro de una persona, o null si no hay nada que mostrar.
    // chalecoDecidido: estado de la ventana del chaleco, para cuando este frame no ve el torso
    // (cara muy cerca de la cámara). Pasarlo solo con una persona a la vista.
    function etiqueta(c, minConf, chalecoDecidido = null) {
        const e = evaluar(c, minConf);
        if (!e.chaleco) e.chaleco = chalecoDecidido;
        if (!e.casco && !e.chaleco) return null;
        const partes = [];
        if (e.casco && e.chaleco === 'ok' && e.casco === 'ok') partes.push('CASCO Y CHALECO');
        else {
            if (e.casco) partes.push(e.casco === 'ok' ? 'CON CASCO' : 'SIN CASCO');
            if (e.chaleco) partes.push(e.chaleco === 'ok' ? 'CON CHALECO' : 'SIN CHALECO');
        }
        const texto = (c.nombre ? c.nombre + ' · ' : '') + partes.join(' · ');
        const falta = e.casco === 'falta' || e.chaleco === 'falta';
        return { texto, color: falta ? '#EF4444' : '#10B981' };
    }

    // Voto del frame para un elemento: la peor situación entre las personas a la vista
    function voto(caras, elemento, minConf) {
        const estados = caras.map(c => evaluar(c, minConf)[elemento]);
        return estados.includes('falta') ? 'falta' : estados.includes('ok') ? 'ok' : null;
    }

    // Ventana de votos de un elemento. votar() devuelve el nuevo estado decidido
    // ('ok' / 'falta') cuando cambia, o null si sigue igual o todavía no hay consenso.
    function ventana() {
        let votos = [], estado = null;
        return {
            get estado() { return estado; },
            get progreso() {
                if (!votos.length) return 0;
                const n = v => votos.filter(x => x === v).length / votos.length;
                return Math.max(n('falta'), n('ok'));
            },
            votar(v) {
                if (!v) return null;
                votos.push(v); if (votos.length > WINDOW_SIZE) votos.shift();
                const ratio = x => votos.filter(y => y === x).length / votos.length;
                const decidido = ratio('falta') >= THRESH_FALTA ? 'falta'
                               : ratio('ok') >= THRESH_OK && votos.length >= WINDOW_SIZE ? 'ok' : null;
                if (!decidido || decidido === estado) return null;
                estado = decidido;
                return decidido;
            },
            reset() { votos = []; estado = null; },
        };
    }

    // Seguimiento de una cámara: una ventana por elemento activo.
    // Si la cámara queda vacía, los verdes dejan de valer; las alertas siguen hasta ver a alguien con EPP.
    function seguimiento() {
        const FRAMES_VACIO_RESET = 15;
        const ventanas = { casco: ventana(), chaleco: ventana() };
        let modos = { casco: true, chaleco: true }, vacios = 0;
        return {
            ventanas,
            get modos() { return modos; },
            // Devuelve [{elemento, estado}] con los cambios de este frame
            procesar(data, minConf) {
                if (data.modos) modos = data.modos;
                const caras = data.faces || [];
                const activos = Object.keys(ventanas).filter(k => modos[k]);
                for (const k of Object.keys(ventanas)) if (!modos[k]) ventanas[k].reset();
                const votos = Object.fromEntries(activos.map(k => [k, voto(caras, k, minConf)]));
                if (!activos.some(k => votos[k])) {
                    vacios++;
                    if (vacios >= FRAMES_VACIO_RESET)
                        activos.forEach(k => { if (ventanas[k].estado === 'ok') ventanas[k].reset(); });
                    return [];
                }
                vacios = 0;
                const cambios = [];
                for (const k of activos) {
                    const nuevo = ventanas[k].votar(votos[k]);
                    if (nuevo) cambios.push({ elemento: k, estado: nuevo });
                }
                return cambios;
            },
            // Resumen para el panel: clase 'ok' / 'alerta' / null, título y detalle por elemento
            resumen() {
                const activos = Object.keys(ventanas).filter(k => modos[k]);
                const faltan = activos.filter(k => ventanas[k].estado === 'falta');
                const ok = activos.filter(k => ventanas[k].estado === 'ok');
                const detalle = activos.map(k => {
                    const e = ventanas[k].estado;
                    const t = e === 'ok' ? 'OK' : e === 'falta' ? 'FALTA'
                            : `verificando (${Math.round(ventanas[k].progreso * 100)} %)`;
                    return `${k === 'casco' ? 'Casco' : 'Chaleco'}: ${t}`;
                }).join(' · ');
                if (faltan.length)
                    return { clase: 'alerta', titulo: 'ALERTA: SIN ' + faltan.map(k => NOMBRES[k].toUpperCase()).join(' NI '), detalle };
                if (ok.length === activos.length && activos.length)
                    return { clase: 'ok', titulo: activos.length > 1 ? 'TODOS CON EPP' : `TODOS CON ${activos[0].toUpperCase()}`, detalle };
                return { clase: null, titulo: null, detalle };
            },
            // Versión corta para chips: "CASCO OK · SIN CHALECO"
            corto() {
                return Object.keys(ventanas).filter(k => modos[k]).map(k => {
                    const e = ventanas[k].estado, K = k.toUpperCase();
                    return e === 'ok' ? `${K} OK` : e === 'falta' ? `SIN ${K}` : `${K} …`;
                }).join(' · ');
            },
            // Estado para la alarma física: 'sin_casco' / 'sin_chaleco' (rojo), 'con_casco' (verde) o null
            alarma() {
                const r = this.resumen();
                if (r.clase === 'alerta') return ventanas.casco.estado === 'falta' && modos.casco ? 'sin_casco' : 'sin_chaleco';
                return r.clase === 'ok' ? 'con_casco' : null;
            },
            reset() { ventanas.casco.reset(); ventanas.chaleco.reset(); vacios = 0; },
        };
    }

    // Quiénes tienen el faltante de un elemento (para la alerta)
    function enFalta(caras, elemento, minConf) {
        const nombres = caras.filter(c => evaluar(c, minConf)[elemento] === 'falta').map(c => c.nombre || 'Sin identificar');
        return [...new Set(nombres)].join(', ');
    }

    // Confianza de referencia para el registro de una alerta o un OK
    function confianza(caras, elemento, estado, minConf) {
        const c = caras.find(x => evaluar(x, minConf)[elemento] === estado);
        // El chaleco no trae probabilidad (modelo + regla de color): se registra como 100
        if (!c || elemento !== 'casco') return 100;
        return Math.max(c.con_casco, c.sin_casco);
    }

    return { etiqueta, seguimiento, enFalta, confianza, evaluar };
})();
