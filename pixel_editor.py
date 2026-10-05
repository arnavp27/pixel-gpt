import streamlit as st


HTML = """
    <div class="editor">
        <div class="toolbar">
            <h3>Draw the left half</h3>
            <button class="clear" type="button">Clear</button>
        </div>
        <div class="canvas">
            <div class="halves"><span>You draw</span><span>Model fills</span></div>
            <div class="pixels" role="group" aria-label="Draw on the left half; the model fills the right"></div>
        </div>
        <p class="hint">Click / drag to draw or erase. Blank is white.</p>
    </div>
    """

CSS = """
    .editor { font: 12px var(--st-font, monospace); color: #243128; }
    .toolbar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; gap: 6px; }
    .toolbar h3 { margin: 0; font-size: 16px; font-weight: 600; }
    .toolbar button { font: inherit; color: inherit; background: #fffef9; cursor: pointer;
        border: 1px solid #bbc7b5; border-radius: 5px; padding: 8px 9px; }
    .toolbar button:hover { border-color: #395f3b; }
    button:focus-visible { outline: 2px solid #ad7b32; outline-offset: 2px; }
    .canvas { width: min(100%, 288px, calc(100dvh - 445px)); margin: auto; }
    .halves { display: grid; grid-template-columns: 1fr 1fr; text-align: center;
        margin-bottom: 6px; color: #526056; font-size: 11px; }
    .halves span:first-child { color: #395f3b; font-weight: bold; }
    .pixels { display: grid; grid-template-columns: repeat(8, 1fr); gap: 2px;
        touch-action: none; user-select: none; }
    .pixel { aspect-ratio: 1; padding: 0; border: 1px solid #b7c2b1;
        border-radius: 2px; background: white; cursor: crosshair; }
    .pixel:disabled { background: repeating-linear-gradient(135deg, #e2e5dd 0 3px, #d2d8cb 3px 6px);
        cursor: default; opacity: 1; }
    .pixel.ink { background: #243128; border-color: #243128; }
    .pixel:enabled:hover { border-color: #395f3b; box-shadow: inset 0 0 0 1px #395f3b; }
    .pixel:focus-visible { outline: 2px solid #b77725; outline-offset: 1px; z-index: 1; }
    .hint { margin: 12px 0 0; color: #526056; line-height: 1.5; }
    @media (max-width: 640px) {
        .canvas { width: min(100%, 168px); }
        .toolbar { margin-bottom: 8px; }
        .toolbar h3 { font-size: 14px; }
        .toolbar button { padding: 6px 9px; }
        .hint { margin-top: 8px; font-size: 11px; }
    }
    """

JS = """
    export default function({ parentElement, data, setStateValue }) {
        const grid = parentElement.querySelector('.pixels');
        const pixels = [...data.pixels];
        let ink = 1;
        let painting = false;
        const focused = parentElement.activeElement?.dataset.index;

        const buttons = pixels.map((_, i) => {
            const button = document.createElement('button');
            button.type = 'button';
            button.dataset.index = i;
            button.disabled = i % 8 >= 4;
            return button;
        });
        grid.replaceChildren(...buttons);
        if (focused !== undefined) buttons[Number(focused)].focus({preventScroll: true});

        function render() {
            buttons.forEach((button, i) => {
                button.className = 'pixel' + (pixels[i] === 1 ? ' ink' : '');
                if (!button.disabled) button.setAttribute('aria-pressed', String(pixels[i] === 1));
                const state = button.disabled ? 'model fills' : pixels[i] ? 'dark' : 'white';
                button.setAttribute('aria-label', `Row ${Math.floor(i / 8) + 1}, column ${i % 8 + 1}: ${state}`);
            });
        }

        function publish() {
            setStateValue('pixels', [...pixels]);
        }

        function paint(index) {
            if (index % 8 >= 4) return;
            pixels[index] = ink;
            render();
        }

        grid.onpointerdown = event => {
            if (event.button !== 0 || event.target.dataset.index === undefined) return;
            event.preventDefault();
            const index = Number(event.target.dataset.index);
            if (index % 8 >= 4) return;
            ink = pixels[index] === 1 ? 0 : 1;
            painting = true;
            grid.setPointerCapture(event.pointerId);
            paint(index);
        };
        grid.onpointermove = event => {
            if (!painting) return;
            const rect = grid.getBoundingClientRect();
            const x = Math.floor((event.clientX - rect.left) / rect.width * 8);
            const y = Math.floor((event.clientY - rect.top) / rect.height * 8);
            if (x >= 0 && x < 4 && y >= 0 && y < 8) paint(y * 8 + x);
        };
        grid.onpointerup = grid.onpointercancel = grid.onlostpointercapture = () => {
            if (painting) publish();
            painting = false;
        };
        grid.onclick = event => {
            if (event.detail !== 0 || event.target.dataset.index === undefined) return;
            const index = Number(event.target.dataset.index);
            ink = pixels[index] === 1 ? 0 : 1;
            paint(index);
            publish();
        };
        grid.onkeydown = event => {
            const moves = {ArrowLeft: [0, -1], ArrowRight: [0, 1], ArrowUp: [-1, 0], ArrowDown: [1, 0]};
            if (moves[event.key] === undefined || event.target.dataset.index === undefined) return;
            event.preventDefault();
            const index = Number(event.target.dataset.index);
            const [dy, dx] = moves[event.key];
            const row = Math.max(0, Math.min(7, Math.floor(index / 8) + dy));
            const column = Math.max(0, Math.min(3, index % 8 + dx));
            buttons[row * 8 + column].focus();
        };
        parentElement.querySelector('.clear').onclick = () => {
            pixels.forEach((_, i) => { pixels[i] = i % 8 < 4 ? 0 : null; });
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
    pixels = [int(pixel == 1) if i % 8 < 4 else None for i, pixel in enumerate(pixels)]
    values = {"pixels": pixels}
    result = editor(data=values, default=values, on_pixels_change=lambda: None, key=key)
    fixed = [int(pixel == 1) if i % 8 < 4 else None for i, pixel in enumerate(result.pixels)]
    st.session_state["drawing_pixels"] = fixed
    return fixed
