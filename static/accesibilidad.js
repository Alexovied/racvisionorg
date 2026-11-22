// Panel de accesibilidad compartido. Se carga en el <head> sin defer: aplica las
// preferencias guardadas antes del primer pintado y arma el panel al cargar el DOM.
// Las preferencias viven en localStorage ("rac-a11y") y se avisan con el evento "rac-a11y".
(function () {
    const CLAVE = 'rac-a11y';
    const html = document.documentElement;
    const base = { texto: 0, contraste: false, subrayado: false, espaciado: false, quieto: false };
    let prefs = Object.assign({}, base);
    try { prefs = Object.assign(prefs, JSON.parse(localStorage.getItem(CLAVE) || '{}')); } catch (e) {}

    function aplicar() {
        html.classList.remove('a11y-texto-1', 'a11y-texto-2');
        if (prefs.texto > 0) html.classList.add('a11y-texto-' + prefs.texto);
        ['contraste', 'subrayado', 'espaciado', 'quieto'].forEach(k => html.classList.toggle('a11y-' + k, !!prefs[k]));
        window.racA11y = prefs;
        document.dispatchEvent(new CustomEvent('rac-a11y', { detail: prefs }));
    }
    function guardar() {
        try { localStorage.setItem(CLAVE, JSON.stringify(prefs)); } catch (e) {}
        aplicar();
    }
    aplicar();

    const TXT = {
        es: { abrir: 'Opciones de accesibilidad', titulo: 'Accesibilidad', ayuda: 'Ajustá la página a tu comodidad. Se recuerda en este navegador.',
              tamano: 'Tamaño del texto', normal: 'Texto normal', grande: 'Texto grande', mayor: 'Texto más grande',
              contraste: 'Alto contraste', contrasteAyuda: 'Textos más claros y fondos más oscuros',
              subrayado: 'Subrayar enlaces', subrayadoAyuda: 'Distinguir los enlaces sin depender del color',
              espaciado: 'Más espacio en el texto', espaciadoAyuda: 'Ayuda a leer con dislexia o baja visión',
              quieto: 'Sin animaciones', quietoAyuda: 'Quita movimientos y efectos de scroll',
              reset: 'Restablecer', cerrar: 'Cerrar', saltar: 'Saltar al contenido' },
        en: { abrir: 'Accessibility options', titulo: 'Accessibility', ayuda: 'Adjust the page to suit you. It is remembered in this browser.',
              tamano: 'Text size', normal: 'Normal text', grande: 'Large text', mayor: 'Larger text',
              contraste: 'High contrast', contrasteAyuda: 'Brighter text and darker backgrounds',
              subrayado: 'Underline links', subrayadoAyuda: 'Spot links without relying on color',
              espaciado: 'More text spacing', espaciadoAyuda: 'Helps with dyslexia or low vision',
              quieto: 'No animations', quietoAyuda: 'Removes motion and scroll effects',
              reset: 'Reset', cerrar: 'Close', saltar: 'Skip to content' },
    };

    document.addEventListener('DOMContentLoaded', () => {
        const t = TXT[(html.lang || 'es').slice(0, 2) === 'en' ? 'en' : 'es'];

        // "Saltar al contenido" como primer elemento enfocable
        const destino = document.getElementById('contenido') || document.querySelector('main');
        if (destino) {
            if (!destino.id) destino.id = 'contenido';
            if (!destino.hasAttribute('tabindex')) destino.setAttribute('tabindex', '-1');
            const saltar = document.createElement('a');
            saltar.className = 'a11y-saltar'; saltar.href = '#' + destino.id; saltar.textContent = t.saltar;
            saltar.addEventListener('click', e => { e.preventDefault(); destino.focus(); destino.scrollIntoView(); });
            document.body.prepend(saltar);
        }

        const boton = document.createElement('button');
        boton.type = 'button'; boton.className = 'a11y-boton';
        boton.setAttribute('aria-label', t.abrir); boton.setAttribute('aria-expanded', 'false'); boton.setAttribute('aria-controls', 'a11y-panel');
        boton.innerHTML = '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><circle cx="12" cy="4" r="2"/><path d="M19 8.5c-2.2.6-4.6.9-7 .9s-4.8-.3-7-.9l-.5 1.9c1.9.5 3.9.8 6 .9V14l-2 7h2.2l1.8-5.6 1.8 5.6h2.2l-2-7v-2.7c2.1-.1 4.1-.4 6-.9z"/></svg>';

        const opcion = (k, ayuda) => `<label class="a11y-opcion"><span>${t[k]}<small>${t[ayuda]}</small></span><input type="checkbox" role="switch" data-k="${k}"></label>`;
        const panel = document.createElement('div');
        panel.id = 'a11y-panel'; panel.className = 'a11y-panel'; panel.hidden = true;
        panel.setAttribute('role', 'dialog'); panel.setAttribute('aria-labelledby', 'a11y-titulo');
        panel.innerHTML = `
            <h2 id="a11y-titulo">${t.titulo}</h2>
            <p class="a11y-ayuda">${t.ayuda}</p>
            <fieldset><legend>${t.tamano}</legend><div class="a11y-tamanos">
                <label><input type="radio" name="a11y-texto" value="0"><span aria-hidden="true">A</span><span class="a11y-oculto">${t.normal}</span></label>
                <label><input type="radio" name="a11y-texto" value="1"><span aria-hidden="true">A+</span><span class="a11y-oculto">${t.grande}</span></label>
                <label><input type="radio" name="a11y-texto" value="2"><span aria-hidden="true">A++</span><span class="a11y-oculto">${t.mayor}</span></label>
            </div></fieldset>
            ${opcion('contraste', 'contrasteAyuda')}${opcion('subrayado', 'subrayadoAyuda')}${opcion('espaciado', 'espaciadoAyuda')}${opcion('quieto', 'quietoAyuda')}
            <div class="a11y-acciones"><button type="button" class="a11y-reset">${t.reset}</button><button type="button" class="a11y-cerrar">${t.cerrar}</button></div>`;
        document.body.append(boton, panel);

        const pintar = () => {
            panel.querySelectorAll('input[name="a11y-texto"]').forEach(r => { r.checked = Number(r.value) === prefs.texto; });
            panel.querySelectorAll('input[data-k]').forEach(c => { c.checked = !!prefs[c.dataset.k]; });
        };
        const abrir = () => { pintar(); panel.hidden = false; boton.setAttribute('aria-expanded', 'true'); panel.querySelector('input:checked, input').focus(); };
        const cerrar = () => { panel.hidden = true; boton.setAttribute('aria-expanded', 'false'); boton.focus(); };

        boton.addEventListener('click', () => (panel.hidden ? abrir() : cerrar()));
        panel.querySelector('.a11y-cerrar').addEventListener('click', cerrar);
        panel.querySelector('.a11y-reset').addEventListener('click', () => { prefs = Object.assign({}, base); guardar(); pintar(); });
        panel.addEventListener('change', e => {
            if (e.target.name === 'a11y-texto') prefs.texto = Number(e.target.value);
            else if (e.target.dataset.k) prefs[e.target.dataset.k] = e.target.checked;
            guardar();
        });
        document.addEventListener('keydown', e => { if (e.key === 'Escape' && !panel.hidden) cerrar(); });
        // Hacer clic afuera cierra el panel sin robar el foco
        document.addEventListener('click', e => {
            if (!panel.hidden && !panel.contains(e.target) && !boton.contains(e.target)) { panel.hidden = true; boton.setAttribute('aria-expanded', 'false'); }
        });
    });
})();
