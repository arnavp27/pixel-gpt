import io
import json

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import streamlit as st
import torch

from sprites.data import ROOT, SIZE
from ui.pixel_editor import draw_pixels
from sprites.sample import generate, load_model
from sprites import MODELS

LABELS = {"bigram": "Bigram", "mlp": "MLP", "transformer": "Transformer"}
CONTEXT = {"bigram": "1 previous pixel", "mlp": "4 previous pixels + position",
           "transformer": "All previous pixels + position"}



@st.cache_resource
def cached_model(name, modified):
    return load_model(name)[0]


def make_batch(names, count, fixed, temperature, seed):
    images = {}
    for name in names:
        path = ROOT / "checkpoints" / "sprites" / f"{name}.pt"
        modified = path.stat().st_mtime_ns if path.exists() else 0
        model = cached_model(name, modified)
        images[name] = generate(model, count, temperature=temperature, seed=seed, fixed=fixed)
    return {"images": images, "count": count, "fixed": fixed.copy(),
            "temperature": temperature, "seed": seed}


def make_sheet(images, columns=4):
    columns = min(columns, len(images))
    rows = (len(images) + columns - 1) // columns
    scale, gap = 12, 12
    tile = SIZE * scale
    sheet = Image.new("RGB", (columns * (tile + gap) + gap, rows * (tile + gap) + gap), "#f8f8f2")
    draw = ImageDraw.Draw(sheet)
    for i, image in enumerate(images):
        pixels = image.reshape(SIZE, SIZE).numpy().astype(bool)
        rgb = np.where(pixels[:, :, None], [36, 49, 40], [255, 255, 255]).astype(np.uint8)
        sprite = Image.fromarray(rgb).resize((tile, tile), Image.Resampling.NEAREST)
        x, y = gap + i % columns * (tile + gap), gap + i // columns * (tile + gap)
        sheet.paste(sprite, (x, y))
        draw.rectangle((x - 1, y - 1, x + tile, y + tile), outline="#d6dbcf")
    return sheet


def show_samples(name, images, batch, columns=4):
    comparison = len(batch["images"]) > 1
    if comparison:
        st.subheader(LABELS[name])
        st.markdown(f'<p class="model-note">{CONTEXT[name]}</p>', unsafe_allow_html=True)
    sheet = make_sheet(images, max(columns, len(images) // 4))
    st.image(sheet, width="stretch")
    grids = images.reshape(-1, SIZE, SIZE)
    symmetric = (grids == grids.flip(2)).flatten(1).all(1).sum().item()
    unique = len(torch.unique(images, dim=0))
    st.markdown(f'<div class="sample-stats"><span><b>{symmetric}/{len(images)}</b> mirrored</span>'
                f'<span><b>{unique}</b> unique</span></div>', unsafe_allow_html=True)
    buffer = io.BytesIO()
    sheet.save(buffer, format="PNG")
    st.download_button("↓ PNG" if comparison else "Download PNG", buffer.getvalue(),
                       file_name=f"{name}-{batch['seed']}.png", mime="image/png",
                       key=f"download_{name}", on_click="ignore", width="stretch")


def render(mode):
    if mode in ("Generate", "Compare"):
        st.markdown('<p class="intro">Trained on mirror images. '
                    'Draw the left half; the model predicts the right.</p>', unsafe_allow_html=True)
        with st.container(key="workspace"):
            canvas, output = st.columns([1, 1.8], gap="medium")
            with canvas, st.container(border=True, key="canvas"):
                fixed = draw_pixels(key="drawing")

        settings_key = mode.lower() + "_settings"
        settings = st.session_state.get(settings_key, {"model": "transformer", "temperature": 1.0,
                                                       "seed": 123, "count": 16 if mode == "Generate" else 8})
        with st.container(key="sampling"):
            with st.form(f"{mode.lower()}_form"):
                model_col, temp_col, seed_col, count_col, action = st.columns([2, 2.6, 1.3, 1.2, 1.8], gap="small")
                with model_col:
                    if mode == "Generate":
                        name = st.selectbox("Model", list(MODELS), index=list(MODELS).index(settings["model"]),
                                            format_func=LABELS.get)
                        names = [name]
                    else:
                        names = list(MODELS)
                        st.selectbox("Models", ["All three"], disabled=True)
                with temp_col:
                    temperature = st.slider("Temperature", 0.0, 2.0, settings["temperature"], 0.1,
                                            help="0 always picks the most likely pixel. Higher values add randomness.")
                with seed_col:
                    seed = st.number_input("Seed", min_value=0, max_value=2**31 - 1, value=settings["seed"], step=1)
                with count_col:
                    count = st.selectbox("Samples", [4, 8, 16, 32], index=[4, 8, 16, 32].index(settings["count"]))
                with action:
                    submitted = st.form_submit_button("Generate" if mode == "Generate" else "Compare",
                                                      type="primary", width="stretch")
        if submitted:
            st.session_state[settings_key] = {"model": names[0], "temperature": temperature,
                                             "seed": int(seed), "count": count}

        state_key = mode.lower() + "_batch"
        error = None
        if submitted:
            try:
                with output, st.spinner("Sampling pixels…"):
                    st.session_state[state_key] = make_batch(names, count, fixed, temperature, int(seed))
            except (FileNotFoundError, RuntimeError, ValueError) as problem:
                error = str(problem)
        with output, st.container(border=True, key="gallery"):
            st.subheader("Completed patterns" if mode == "Generate" else "Three models, one drawing")
            if error:
                st.error(error)
                if state_key in st.session_state:
                    st.caption("The previous samples are still shown below.")
            if state_key in st.session_state:
                batch = st.session_state[state_key]
                model_label = LABELS[next(iter(batch["images"]))] + " · " if mode == "Generate" else ""
                changed = fixed != batch["fixed"]
                note = f'{model_label}Seed {batch["seed"]} · Temp {batch["temperature"]:.1f}'
                if changed:
                    note = f"Drawing changed · {mode.lower()} again to apply it"
                st.markdown(f'<p class="batch-note{" changed" if changed else ""}">{note}</p>',
                            unsafe_allow_html=True)
                if mode == "Generate":
                    for name, images in batch["images"].items():
                        show_samples(name, images, batch)
                else:
                    with st.container(key="comparison"):
                        for column, (name, images) in zip(st.columns(3), batch["images"].items()):
                            with column:
                                show_samples(name, images, batch, columns=2)
            elif not error:
                st.markdown(f'<div class="empty-gallery"><p>Your pattern starts here.</p>'
                            f'<span>Draw on the left, then press {mode}. '
                            'Your half stays the same in every sample.</span></div>', unsafe_allow_html=True)
    else:
        path = ROOT / "results" / "sprites" / "metrics.json"
        if not path.exists():
            st.info("Run python -m sprites.evaluate to save the model comparison.")
        else:
            report = json.loads(path.read_text())
            st.subheader("What the models learned")
            st.caption("The models learned from mirror images. Better predictions mean lower loss.")
            scores = {"Measure": ["Parameters", "Test loss ↓", "Mirrored pairs", "Symmetric sprites"]}
            for name, metrics in report["models"].items():
                symmetric = round(metrics["symmetric_images"] * report["sample_count"])
                scores[LABELS[name]] = [f'{metrics["parameters"]:,}', f'{metrics["test_loss"]:.4f}',
                                        f'{metrics["matching_pairs"]:.2%}', f'{symmetric} / {report["sample_count"]}']
            with st.container(key="results-panels"):
                table, learning = st.columns([1, 1], gap="medium")
            with table, st.container(border=True, key="scores"):
                st.subheader("Model comparison")
                st.table(pd.DataFrame(scores).set_index("Measure"))
                st.caption("Loss: lower is better. Symmetry: the left and right halves are mirror images.")
            histories = []
            for name in MODELS:
                path = ROOT / "results" / "sprites" / f"{name}_loss.csv"
                if path.exists():
                    history = pd.read_csv(path).set_index("step")["val_loss"].rename(LABELS[name])
                    histories.append(history)
            if histories:
                with learning, st.container(border=True, key="learning"):
                    st.subheader("Learning curves")
                    st.caption("Validation loss · lower is better")
                    st.line_chart(pd.concat(histories, axis=1), x_label="Training step", y_label="Loss",
                                  color=["#748578", "#ad7b32", "#315caa"][:len(histories)], height=240)
            with st.container(key="results-footer"):
                note, download = st.columns([2, 1], vertical_alignment="center")
                with note:
                    st.caption(f"{report['split_sizes']['test']} test images · {report['sample_count']} samples/model.")
                with download:
                    st.download_button("↓ JSON", json.dumps(report, indent=2), file_name="metrics.json",
                                       help="Download the saved evaluation metrics",
                                       mime="application/json", on_click="ignore", width="stretch")
