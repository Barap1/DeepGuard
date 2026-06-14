import unittest

import torch

from model import DeepfakeDetector


class DeepfakeDetectorShapeTest(unittest.TestCase):
    def test_forward_output_shape(self):
        model = DeepfakeDetector(pretrained=False)
        model.eval()

        inputs = torch.randn(1, 2, 3, 224, 224)

        with torch.no_grad():
            output = model(inputs)

        self.assertEqual(tuple(output.shape), (1, 1))


if __name__ == "__main__":
    unittest.main()
