import unittest

import torch

from data import START, inputs_for, make_dataset
from sample import generate
from train import MODELS


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


if __name__ == "__main__":
    unittest.main()
