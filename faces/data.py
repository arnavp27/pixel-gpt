import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import time
import zipfile

from PIL import Image
import torch

from faces import ROOT
from faces.preprocess import encode

DIRECTORY = ROOT / "data" / "ffhq"
DATA_FILE = ROOT / "data" / "faces.pt"
ARCHIVE_URL = ("https://huggingface.co/datasets/nuwandaa/ffhq128/resolve/"
               "1b277b6efe3ba86ff0c712b88753892ca5c4e7a3/thumbnails128x128.zip")
ARCHIVE_SHA = "ef9fd4b39922cf599c6446e45b03b334824c6ce19b4b8bdf7aee6d06df355f3f"
METADATA_URL = ("https://drive.usercontent.google.com/download?"
                "id=16N0RV4fHI6joBuKbQAoG34V_cQk7vxSA&export=download&confirm=t")
METADATA_MD5 = "425ae20f06a4da1d4dc0f46d40ba5fd6"


def download(url, path, expected, algorithm="sha256"):
    import requests

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open("rb") as file:
            if hashlib.file_digest(file, algorithm).hexdigest() == expected:
                return
        raise ValueError(f"Checksum mismatch: {path}")
    partial = path.with_suffix(path.suffix + ".part")
    for attempt in range(3):
        offset = partial.stat().st_size if partial.exists() else 0
        try:
            with requests.get(url, headers={"Range": f"bytes={offset}-"} if offset else {},
                              stream=True, timeout=(30, 90)) as response:
                response.raise_for_status()
                append = offset > 0 and response.status_code == 206
                if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError("Unexpected download range")
                written = offset if append else 0
                report_at = written + 100 * 1024 ** 2
                with partial.open("ab" if append else "wb") as file:
                    for chunk in response.iter_content(4 * 1024 ** 2):
                        file.write(chunk)
                        written += len(chunk)
                        if written >= report_at:
                            print(f"{path.name}: {written / 1024**2:.0f} MB", flush=True)
                            report_at = written + 100 * 1024 ** 2
            break
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    with partial.open("rb") as file:
        if hashlib.file_digest(file, algorithm).hexdigest() != expected:
            raise ValueError(f"Checksum mismatch: {partial}")
    partial.replace(path)
    print(f"Downloaded {path.name}", flush=True)


def prepare(size=32, levels=16, seed=42, output=DATA_FILE):
    archive_path = DIRECTORY / "thumbnails128x128.zip"
    metadata_path = DIRECTORY / "ffhq-dataset-v2.json"
    if not archive_path.exists():
        os.environ.setdefault("HF_HOME", str(DIRECTORY / ".cache"))
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        from huggingface_hub import hf_hub_download
        hf_hub_download("nuwandaa/ffhq128", "thumbnails128x128.zip", repo_type="dataset",
                        revision="1b277b6efe3ba86ff0c712b88753892ca5c4e7a3",
                        local_dir=DIRECTORY, token=False)
    download(ARCHIVE_URL, archive_path, ARCHIVE_SHA)
    download(METADATA_URL, metadata_path, METADATA_MD5, "md5")
    metadata = json.loads(metadata_path.read_text())
    images = torch.empty((70000, size * size), dtype=torch.uint8)
    with zipfile.ZipFile(archive_path) as archive:
        files = {int(Path(name).stem): name for name in archive.namelist()
                 if name.lower().endswith(".png") and Path(name).stem.isdecimal()}
        if set(files) != set(range(70000)):
            raise ValueError("Expected all 70,000 FFHQ thumbnails")
        for i in range(70000):
            content = archive.read(files[i])
            if hashlib.md5(content).hexdigest() != metadata[str(i)]["thumbnail"]["file_md5"]:
                raise ValueError(f"Thumbnail {i} differs from the official FFHQ metadata")
            with Image.open(io.BytesIO(content)) as image:
                images[i] = encode(image, size, levels)
            if (i + 1) % 10000 == 0:
                print(f"Prepared {i + 1}/70000", flush=True)
    train_ids = torch.tensor([i for i in range(70000) if metadata[str(i)]["category"] == "training"])
    held_out = torch.tensor([i for i in range(70000) if metadata[str(i)]["category"] == "validation"])
    rng = torch.Generator().manual_seed(seed)
    held_out = held_out[torch.randperm(len(held_out), generator=rng)]
    split_ids = {"train": train_ids, "val": held_out[:5000], "test": held_out[5000:]}
    if [len(split_ids[k]) for k in ("train", "val", "test")] != [60000, 5000, 5000]:
        raise ValueError("Unexpected FFHQ split sizes")
    digest = hashlib.sha256(images.numpy().tobytes())
    digest.update(f"{size}:{levels}:{seed}:left_then_right".encode())
    data = {split: images[ids] for split, ids in split_ids.items()}
    data.update({"ids": split_ids, "size": size, "levels": levels, "seed": seed,
                 "order": "left_then_right", "id": digest.hexdigest(), "source": "FFHQ v2 thumbnails"})
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output)
    print(f"Saved {output}", flush=True)


def load_data(path=DATA_FILE):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data["order"] != "left_then_right":
        raise ValueError("Unsupported pixel order")
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--levels", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DATA_FILE)
    args = parser.parse_args()
    if args.size < 4 or args.size % 2 or not 2 <= args.levels <= 256:
        parser.error("Use an even size and 2 to 256 gray levels")
    torch.set_num_threads(2)
    prepare(args.size, args.levels, args.seed, args.output)


if __name__ == "__main__":
    main()
