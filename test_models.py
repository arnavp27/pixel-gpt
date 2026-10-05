import unittest

import numpy as np
from PIL import Image

import torch

from sprites.data import START, inputs_for, make_dataset
from sprites.sample import generate
from sprites import MODELS


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_data_split(self):
        data = make_dataset(200)
        images = torch.cat([data[key] for key in ("train", "val", "test")])
        self.assertEqual(len(torch.unique(images, dim=0)), 200)
        grids = images.view(-1, 8, 8)
        self.assertTrue(torch.equal(grids[:, :, :4], grids[:, :, 4:].flip(2)))

    def test_input_target_shift(self):
        images = torch.tensor([[0, 1, 1, 0], [1, 0, 0, 1]])
        expected = torch.tensor([[START, 0, 1, 1], [START, 1, 0, 0]])
        self.assertTrue(torch.equal(inputs_for(images), expected))

    def test_no_future_leakage(self):
        torch.manual_seed(42)
        x = inputs_for(torch.randint(0, 2, (2, 64)))
        for name, constructor in MODELS.items():
            with self.subTest(model=name):
                model = constructor().eval()
                full = model(x)
                self.assertEqual(tuple(full.shape), (2, 64, 2))
                changed = x.clone()
                changed[:, 20:] = 1 - changed[:, 20:]
                torch.testing.assert_close(model(changed)[:, :20], full[:, :20])

    def test_sampling(self):
        fixed = [None] * 64
        fixed[0], fixed[10], fixed[40], fixed[63] = 1, 0, 1, 0
        for name, constructor in MODELS.items():
            with self.subTest(model=name):
                model = constructor()
                first = generate(model, 2, fixed=fixed, seed=9)
                second = generate(model, 2, fixed=fixed, seed=9)
                self.assertTrue(torch.equal(first, second))
                self.assertEqual(tuple(first.shape), (2, 64))
                self.assertTrue(torch.all((first == 0) | (first == 1)))
                expected = torch.tensor([1, 0, 1, 0]).expand(2, -1)
                self.assertTrue(torch.equal(first[:, [0, 10, 40, 63]], expected))
                self.assertEqual(generate(model, 1, prefix="0101")[0, :4].tolist(), [0, 1, 0, 1])
                self.assertTrue(torch.equal(generate(model, 1, temperature=0, seed=1),
                                            generate(model, 1, temperature=0, seed=2)))

    def test_fixed_pixels_become_context(self):
        model = MODELS["bigram"]()
        with torch.no_grad():
            model.logits.copy_(torch.tensor([[9., -9.], [-9., 9.], [9., -9.]]))
        fixed = [None] * 64
        fixed[10], fixed[22] = 1, 0
        samples = generate(model, 1, temperature=0, fixed=fixed)
        self.assertEqual(samples[0].tolist(), [0] * 10 + [1] * 12 + [0] * 42)


class FaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self):
        from faces.model import Config, Transformer
        torch.manual_seed(42)
        return Transformer(Config(size=8, width=32, heads=4, layers=2, dropout=0)).eval()

    def test_pixel_order_and_roundtrip(self):
        from faces.preprocess import encode, decode, inputs_for
        pixels = np.arange(16, dtype=np.uint8).reshape(4, 4)
        image = Image.fromarray(pixels * 17)
        tokens = encode(image, size=4)
        self.assertEqual(tokens.tolist(), [0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15])
        np.testing.assert_array_equal(np.asarray(decode(tokens, size=4)), np.asarray(image))
        self.assertEqual(inputs_for(tokens[None])[0].tolist(), [16] + tokens[:-1].tolist())

    def test_causality_and_cache(self):
        from faces.preprocess import inputs_for
        model = self.model()
        x = inputs_for(torch.randint(16, (2, 64)))
        with torch.no_grad():
            full = model(x)
            changed = x.clone()
            changed[:, 35:] = 15 - changed[:, 35:]
            torch.testing.assert_close(model(changed)[:, :35], full[:, :35])
            _, cache = model(x[:, :33], use_cache=True)
            single, cache = model(x[:, 33:34], cache=cache, use_cache=True)
            torch.testing.assert_close(single, full[:, 33:34], atol=1e-6, rtol=1e-5)
            chunk, _ = model(x[:, 34:40], cache=cache, use_cache=True)
            torch.testing.assert_close(chunk, full[:, 34:40], atol=1e-6, rtol=1e-5)

    def test_cached_sampling(self):
        from faces.sample import generate as complete
        model = self.model()
        left = torch.randint(16, (32,))
        first = complete(model, left, count=2, seed=9)
        self.assertTrue(torch.equal(first[:, :32], left.expand(2, -1)))
        self.assertTrue(torch.equal(first, complete(model, left, count=2, seed=9)))
        expected = torch.cat((torch.full((1, 1), 16), left[None]), dim=1)
        with torch.no_grad():
            for _ in range(32):
                pixel = model(expected)[:, -1].argmax(-1, keepdim=True)
                expected = torch.cat((expected, pixel), dim=1)
        self.assertTrue(torch.equal(complete(model, left, temperature=0), expected[:, 1:]))
        with self.assertRaises(ValueError):
            complete(model, torch.cat((left, left)))

    def test_loss_only_scores_the_missing_half(self):
        from faces.train import right_loss
        logits = torch.randn(2, 64, 16, requires_grad=True)
        targets = torch.randint(16, (2, 64))
        loss = right_loss(logits, targets)
        changed = targets.clone()
        changed[:, :32] = 15 - changed[:, :32]
        torch.testing.assert_close(loss, right_loss(logits, changed))
        loss.backward()
        self.assertEqual(logits.grad[:, :32].count_nonzero().item(), 0)
        self.assertGreater(logits.grad[:, 32:].abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
