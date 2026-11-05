/* ══════════════════════════════════════════════════════════════════
   War Advisor — Manuale di gioco

   Tre cose sole: chiudere la scheda, muoversi fra i capitoli senza
   riempire la cronologia e ingrandire gli screenshot.
   ══════════════════════════════════════════════════════════════════ */

(function () {
    'use strict';

    /* Il manuale si apre dal campo di battaglia in una scheda nuova. Chiuderla
       è il modo giusto per tornare indietro: ricaricare la pagina della
       battaglia farebbe ricominciare la partita. Il browser lascia chiudere da
       script solo una scheda con una sola voce di cronologia, ed è per questo
       che i link dell'indice non la allungano (vedi sotto). */
    const chiudi = document.getElementById('chiudiManuale');
    const avviso = document.getElementById('avvisoChiusura');
    if (chiudi) {
        chiudi.addEventListener('click', () => {
            window.close();
            // Se la scheda è ancora qui il browser ha detto di no: si spiega
            // cosa fare invece di lasciare un pulsante che non risponde.
            setTimeout(() => { if (avviso) avviso.hidden = false; }, 400);
        });
    }

    /* Link interni: scorrimento senza aggiungere voci alla cronologia. */
    document.addEventListener('click', (event) => {
        const link = event.target.closest('a[href^="#"]');
        if (!link) return;
        const meta = document.getElementById(link.getAttribute('href').slice(1));
        if (!meta) return;
        event.preventDefault();
        meta.scrollIntoView({ behavior: 'smooth', block: 'start' });
        history.replaceState(null, '', `#${meta.id}`);
    });

    /* Capitolo corrente evidenziato nell'indice. */
    const vociIndice = new Map(
        [...document.querySelectorAll('.toc a[href^="#"]')].map(a => [a.getAttribute('href').slice(1), a])
    );
    if ('IntersectionObserver' in window && vociIndice.size) {
        const osservatore = new IntersectionObserver((voci) => {
            voci.forEach((voce) => {
                if (!voce.isIntersecting) return;
                vociIndice.forEach(a => a.classList.remove('is-current'));
                const attiva = vociIndice.get(voce.target.id);
                if (attiva) attiva.classList.add('is-current');
            });
        }, { rootMargin: '-30% 0px -60% 0px' });
        document.querySelectorAll('main section[id]').forEach(s => osservatore.observe(s));
    }

    /* Ingrandimento: un clic sull'immagine la apre a tutto schermo. */
    const zoom = document.getElementById('zoom');
    const zoomImg = document.getElementById('zoomImg');
    const chiudiZoom = () => { zoom.hidden = true; zoomImg.removeAttribute('src'); };
    document.querySelectorAll('figure img').forEach((img) => {
        img.addEventListener('click', () => {
            zoomImg.src = img.currentSrc || img.src;
            zoomImg.alt = img.alt;
            zoom.hidden = false;
        });
    });
    if (zoom) zoom.addEventListener('click', chiudiZoom);
    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape' && zoom && !zoom.hidden) chiudiZoom();
    });
})();
