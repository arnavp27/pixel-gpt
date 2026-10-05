import hashlib
import io
import json

import pandas as pd
from PIL import Image, UnidentifiedImageError
import streamlit as st

from faces import ROOT
from faces.preprocess import crop_image, decode, encode, mirror
from faces.sample import CHECKPOINT, generate, load_model

RESULTS = ROOT / "results" / "faces"


@st.cache_resource
def cached_model(modified):
    return load_model()[0]


def upload_changed():
    uploaded = st.session_state.get("face_upload")
    st.session_state["face_source"] = uploaded.getvalue() if uploaded else None


def png(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def show_results():
    path = RESULTS / "metrics.json"
    st.subheader("Learning to complete faces")
    if not path.exists():
        st.info("The face evaluation is still being prepared.")
        return
    report = json.loads(path.read_text())
    st.caption(f"{report['test_count']:,} held-out faces · {report['sample_count']} generated completions · "
               f"{report['size']} × {report['size']} pixels")
    with st.container(key="face-results"):
        scores, curve = st.columns(2)
        with scores, st.container(border=True):
            st.subheader("Model and mirror")
            st.table(pd.DataFrame({"Measure": ["Right-half loss", "Model pixel error", "Mirror pixel error"],
                                   "Value": [f"{report['test_loss']:.4f}", f"{report['model_mae']:.4f}",
                                             f"{report['mirror_mae']:.4f}"]}).set_index("Measure"))
            st.caption("Lower is better. Pixel error uses shades from 0 to 1; it doesn't measure how natural a face looks.")
        with curve, st.container(border=True):
            st.subheader("Training curve")
            history = RESULTS / "loss.csv"
            if history.exists():
                st.line_chart(pd.read_csv(history).set_index("step")[["train_loss", "val_loss"]], height=210)
    st.caption("The model predicts a possible missing half. It cannot know exactly what was hidden.")


def render(mode):
    st.markdown('<p class="intro">Upload a centered face. The model sees the left half and predicts the right, '
                'one gray pixel at a time.</p>', unsafe_allow_html=True)
    if mode == "Results":
        show_results()
        return
    if not CHECKPOINT.exists():
        st.info("The face model is being trained. You can try the Sprites demo in the meantime.")
        return
    try:
        model = cached_model(CHECKPOINT.stat().st_mtime_ns)
    except (ValueError, RuntimeError, OSError) as error:
        st.error(f"The face model couldn't load: {error}")
        return
    c = model.config
    with st.container(key="face-workspace"):
        source, output = st.columns([1, 1.7], gap="medium")
        with source, st.container(border=True, key="face-input"):
            st.subheader("Your photo")
            upload, example = st.columns(2, vertical_alignment="center", gap="small")
            with upload:
                st.file_uploader("Upload a face", type=["png", "jpg", "jpeg", "webp"],
                                 key="face_upload", on_change=upload_changed, label_visibility="collapsed")
            content = st.session_state.get("face_source")
            credits_path = RESULTS / "examples" / "credits.json"
            credits = json.loads(credits_path.read_text()) if credits_path.exists() else {}
            examples = [RESULTS / "examples" / f"{name}.png" for name in credits]
            attribution = None
            if content is None and examples:
                with example:
                    selected = st.session_state.get("selected_face", 3)
                    index = st.selectbox("Try an example", range(len(examples)), index=min(selected, len(examples) - 1),
                                         format_func=lambda i: f"Face {i + 1}", key="face_example",
                                         label_visibility="collapsed")
                st.session_state["selected_face"] = index
                content = examples[index].read_bytes()
                attribution = credits.get(examples[index].stem)
            if content is None:
                st.caption("Choose a photo to begin.")
                return
            try:
                if len(content) > 10 * 1024 ** 2:
                    raise ValueError("Choose an image smaller than 10 MB")
                image = Image.open(io.BytesIO(content))
                if image.width * image.height > 20_000_000:
                    raise ValueError("Choose a photo smaller than 20 megapixels")
                crop = st.session_state.get("face_crop", (1.0, 0.5, 0.5))
                with st.popover("Adjust crop", width="stretch"):
                    zoom = st.slider("Zoom", 1.0, 4.0, crop[0], 0.1, key="face_zoom")
                    x = st.slider("Horizontal", 0.0, 1.0, crop[1], 0.05, key="face_x")
                    y = st.slider("Vertical", 0.0, 1.0, crop[2], 0.05, key="face_y")
                st.session_state["face_crop"] = (zoom, x, y)
                tokens = encode(crop_image(image, zoom, x, y), c.size, c.levels)
            except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as error:
                st.error(str(error))
                return
            original = decode(tokens, c.size, c.levels)
            preview = original.convert("RGB")
            preview.paste("#d8ded2", (c.size // 2, 0, c.size, c.size))
            st.image(preview.resize((256, 256), Image.Resampling.NEAREST), width="stretch")
            st.caption(f"{c.size} × {c.size} pixels · {c.levels} gray shades. Only the left half reaches the model.")
            if attribution:
                st.caption(f"Photo by {attribution['author']}. [Source]({attribution['photo_url']}) · "
                           f"[License]({attribution['license_url']})")
    left = tokens[:c.size * c.size // 2]
    identity = hashlib.sha256(tokens.numpy().tobytes()).hexdigest()
    previous = st.session_state.get("face_batch", {})
    with st.container(key="face-sampling"):
        with st.form("face_form"):
            temperature_col, seed_col, action = st.columns([2, 1, 1.5], gap="small")
            with temperature_col:
                temperature = st.slider("Temperature", 0.0, 1.5, previous.get("temperature", 0.8), 0.1,
                                        help="Lower values favor the most likely shades. Higher values add variation.")
            with seed_col:
                seed = st.number_input("Seed", min_value=0, max_value=2**31 - 1, value=previous.get("seed", 123), step=1)
            with action:
                submitted = st.form_submit_button("Complete", type="primary", width="stretch")
    if submitted:
        with output, st.spinner("Drawing the missing half…"):
            try:
                completion = generate(model, left, temperature=temperature, seed=int(seed))[0]
                st.session_state["face_batch"] = {"tokens": completion, "original": tokens,
                    "identity": identity, "temperature": temperature, "seed": int(seed)}
            except (ValueError, RuntimeError) as error:
                st.error(f"Couldn't complete this face: {error}")
    with output, st.container(border=True, key="face-output"):
        reveal = st.session_state.get("face_reveal", False)
        st.subheader(("Original photo" if reveal else "A possible other half")
                     if mode == "Complete" else "Learned or mirrored?")
        batch = st.session_state.get("face_batch")
        if batch is None or batch["tokens"].numel() != c.size ** 2:
            st.markdown('<div class="empty-gallery"><p>What could the other half look like?</p>'
                        '<span>Choose a photo, then press Complete. Your visible half stays fixed.</span></div>',
                        unsafe_allow_html=True)
            return
        if identity != batch["identity"]:
            st.caption("Photo or crop changed. Complete again to update the result.")
        else:
            st.caption(f"Seed {batch['seed']} · Temperature {batch['temperature']:.1f}")
        completed = decode(batch["tokens"], c.size, c.levels)
        if mode == "Complete":
            picture = decode(batch["original"], c.size, c.levels) if reveal else completed
            st.image(picture.resize((320, 320), Image.Resampling.NEAREST), width="stretch")
        else:
            columns = st.columns(3 if reveal else 2)
            baseline, prediction = columns[:2]
            with baseline:
                st.caption("Mirrored")
                mirrored = mirror(batch["original"][:c.size * c.size // 2], c.size)
                st.image(decode(mirrored, c.size, c.levels).resize((256, 256), Image.Resampling.NEAREST), width="stretch")
            with prediction:
                st.caption("Predicted")
                st.image(completed.resize((256, 256), Image.Resampling.NEAREST), width="stretch")
            if reveal:
                with columns[2]:
                    st.caption("Original")
                    st.image(decode(batch["original"], c.size, c.levels).resize((256, 256), Image.Resampling.NEAREST),
                             width="stretch")
        st.toggle("Reveal the original", key="face_reveal")
        st.download_button("Download completion", png(completed), file_name=f"face-{batch['seed']}.png",
                           mime="image/png", on_click="ignore", width="stretch")
