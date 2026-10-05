# Face data

The face model uses [Flickr-Faces-HQ (FFHQ)](https://github.com/NVlabs/ffhq-dataset)
by Tero Karras, Samuli Laine, and Timo Aila at NVIDIA.

The 128×128 thumbnail archive comes from the
[nuwandaa/ffhq128 mirror](https://huggingface.co/datasets/nuwandaa/ffhq128), pinned
to revision `1b277b6efe3ba86ff0c712b88753892ca5c4e7a3`. The preparation script
checks the archive's SHA-256 and every PNG's MD5 against NVIDIA's original v2
metadata. It converts each image to 64×64 grayscale with 16 shades.

The official 60,000 training images are used for training. The official 10,000
validation images are shuffled with seed 42 and split evenly into validation
and test sets. The test set is kept out of training and checkpoint selection.

The dataset and metadata are licensed under CC BY-NC-SA 4.0. Individual photos
have their own licenses, listed in the original metadata. This project uses
them for a noncommercial learning demo. Example photos are resized and converted
to grayscale; their authors, source links, and licenses are retained in
`results/faces/examples/credits.json`. The comparison grid has matching credits
in `results/faces/credits.json`.

Reference: Karras, Laine, and Aila, *A Style-Based Generator Architecture for
Generative Adversarial Networks*, CVPR 2019.
