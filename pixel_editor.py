import streamlit as st


HTML = """
    <div class="editor">
        <div class="toolbar">
            <div class="brushes" role="group" aria-label="Brush">
                <button type="button" data-brush="dark">Dark</button>
                <button type="button" data-brush="white">White</button>
                <button type="button" data-brush="eraser">Eraser</button>
            </div>
            <button class="clear" type="button">Clear</button>
        </div>
        <div class="pixels" role="group" aria-label="Pixel drawing"></div>
        <p class="legend"><span>Dark + white: fixed</span><span>Grey: model fills</span></p>
        <output aria-live="polite"></output>
    </div>
    """

CSS = """
    .editor { font: 12px var(--st-font, monospace); color: #243128; }
    .toolbar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; gap: 6px; }
    .brushes { display: flex; gap: 4px; }
    .toolbar button { font: inherit; color: inherit; background: #fffef9; cursor: pointer;
        border: 1px solid #bbc7b5; border-radius: 5px; padding: 8px 9px; }
    .brushes button[aria-pressed="true"] { background: #395f3b; color: white; border-color: #395f3b; }
    .toolbar button:hover { border-color: #395f3b; }
    button:focus-visible { outline: 2px solid #ad7b32; outline-offset: 2px; }
    .pixels { display: grid; grid-template-columns: repeat(8, 1fr); gap: 2px;
        width: min(100%, 288px, calc(100dvh - 420px)); margin: auto; touch-action: none; user-select: none; }
    .pixel { aspect-ratio: 1; padding: 0; border: 1px solid #b7c2b1;
        border-radius: 2px; background: white; cursor: crosshair; }
    .pixel.pending { background: repeating-linear-gradient(135deg, #e2e5dd 0 3px, #d2d8cb 3px 6px); }
    .pixel.ink { background: #243128; border-color: #243128; }
    .pixel:hover { border-color: #395f3b; box-shadow: inset 0 0 0 1px #395f3b; }
    .pixel:focus-visible { outline: 2px solid #b77725; outline-offset: 1px; z-index: 1; }
    .legend { display: flex; justify-content: space-between; margin: 12px 0 8px; color: #526056; }
    output { display: block; text-align: center; font-weight: bold; }
    @media (max-width: 640px) {
        .pixels { width: min(100%, 176px); }
        .toolbar { margin-bottom: 8px; }
        .toolbar button { padding: 6px 9px; }
        .legend { margin: 8px 0 6px; font-size: 11px; }
        output { font-size: 11px; }
    }
    """

JS = """
    export default function({ parentElement, data, setStateValue }) {
        const grid = parentElement.querySelector('.pixels');
        const brushes = parentElement.querySelectorAll('[data-brush]');
        const output = parentElement.querySelector('output');
        const pixels = [...data.pixels];
        let brush = data.brush;
        let painting = false;
        const focused = parentElement.activeElement?.dataset.index;

        const buttons = pixels.map((_, i) => {
            const button = document.createElement('button');
            button.type = 'button';
            button.dataset.index = i;
            return button;
        });
        grid.replaceChildren(...buttons);
        if (focused !== undefined) buttons[Number(focused)].focus({preventScroll: true});

        function render() {
            brushes.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.brush === brush)));
            buttons.forEach((button, i) => {
                const pending = pixels[i] === null;
                button.className = 'pixel' + (pending ? ' pending' : pixels[i] ? ' ink' : '');
                button.setAttribute('aria-pressed', String(!pending && pixels[i] === 1));
                const state = pending ? 'model fills' : pixels[i] ? 'dark' : 'white';
                button.setAttribute('aria-label', `Row ${Math.floor(i / 8) + 1}, column ${i % 8 + 1}: ${state}`);
            });
            const fixed = pixels.filter(pixel => pixel !== null).length;
            output.textContent = fixed === 64 ? 'All 64 pixels fixed; nothing left to generate.'
                : `${fixed} fixed · ${64 - fixed} for the model`;
        }

        function publish() {
            setStateValue('pixels', [...pixels]);
        }

        function paint(index) {
            pixels[index] = brush === 'eraser' ? null : brush === 'dark' ? 1 : 0;
            render();
        }

        grid.onpointerdown = event => {
            if (event.button !== 0 || event.target.dataset.index === undefined) return;
            event.preventDefault();
            const index = Number(event.target.dataset.index);
            painting = true;
            grid.setPointerCapture(event.pointerId);
            paint(index);
        };
        grid.onpointermove = event => {
            if (!painting) return;
            const rect = grid.getBoundingClientRect();
            const x = Math.floor((event.clientX - rect.left) / rect.width * 8);
            const y = Math.floor((event.clientY - rect.top) / rect.height * 8);
            if (x >= 0 && x < 8 && y >= 0 && y < 8) paint(y * 8 + x);
        };
        grid.onpointerup = grid.onpointercancel = () => {
            if (painting) publish();
            painting = false;
        };
        grid.onclick = event => {
            if (event.detail !== 0 || event.target.dataset.index === undefined) return;
            const index = Number(event.target.dataset.index);
            paint(index);
            publish();
        };
        grid.onkeydown = event => {
            const moves = {ArrowLeft: -1, ArrowRight: 1, ArrowUp: -8, ArrowDown: 8};
            if (moves[event.key] === undefined || event.target.dataset.index === undefined) return;
            event.preventDefault();
            const index = Math.max(0, Math.min(63, Number(event.target.dataset.index) + moves[event.key]));
            buttons[index].focus();
        };
        brushes.forEach(button => {
            button.onclick = () => {
                brush = button.dataset.brush;
                render();
                setStateValue('brush', brush);
            };
        });
        parentElement.querySelector('.clear').onclick = () => {
            pixels.fill(null);
            render();
            publish();
        };
        render();
    }
    """


def draw_pixels(key):
    editor = st.components.v2.component("pixel_editor", html=HTML, css=CSS, js=JS)
    state = st.session_state.get(key, {})
    pixels = state.get("pixels", st.session_state.get("drawing_pixels", [None] * 64))
    brush = state.get("brush", st.session_state.get("drawing_brush", "dark"))
    values = {"pixels": pixels, "brush": brush}
    result = editor(data=values, default=values, on_pixels_change=lambda: None,
                    on_brush_change=lambda: None, key=key)
    st.session_state["drawing_pixels"] = result.pixels
    st.session_state["drawing_brush"] = result.brush
    return result.pixels
