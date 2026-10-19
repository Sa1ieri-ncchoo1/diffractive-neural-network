import json
import unittest
from pathlib import Path

import torch

from d2nn.core import Diffraction, contrast_loss
from d2nn.cli import build_model, load_checkpoint, loss_and_scores

ROOT = Path(__file__).resolve().parents[1]


class PhysicsAndCheckpointTests(unittest.TestCase):
    def test_zero_distance_recovers_complex_field(self):
        layer = Diffraction(N_pixels=16, distance=0., pixel_size=8e-6)
        field = torch.randn(2,1,16,16,dtype=torch.complex64)
        torch.testing.assert_close(layer(field), field, atol=1e-6, rtol=1e-5)

    def test_saved_checkpoint_rejects_wrong_detector(self):
        c=json.loads((ROOT/'configs/contrast_100.json').read_text())
        model=build_model(c,torch.device('cpu'))
        load_checkpoint(model,ROOT/c['checkpoint'],c)
        c['ring_distance_px']=20
        wrong=build_model(c,torch.device('cpu'))
        with self.assertRaisesRegex(ValueError,'Detector masks'):
            load_checkpoint(wrong,ROOT/c['checkpoint'],c)

    def test_saved_checkpoint_rejects_wrong_physics(self):
        c=json.loads((ROOT/'configs/contrast_100.json').read_text())
        c['wavelength']=266e-9
        wrong=build_model(c,torch.device('cpu'))
        with self.assertRaisesRegex(ValueError,'Physical buffer'):
            load_checkpoint(wrong,ROOT/c['checkpoint'],c)

    def test_losses_backpropagate_to_every_phase_layer(self):
        c=json.loads((ROOT/'configs/contrast_100.json').read_text())
        c.update(n_pixels=32, num_layers=2, distances=[.01]*3, ring_distance_px=10,region_radius_px=2)
        model=build_model(c,torch.device('cpu'))
        x=torch.rand(2,1,32,32).sqrt()
        y=torch.tensor([0,1])
        for loss_name in ['mse','contrast','pixel_mse']:
            model.zero_grad(set_to_none=True)
            loss, scores=loss_and_scores(model,x,y,loss_name)
            self.assertEqual(scores.shape,(2,10))
            loss.backward()
            for phase in model.d2nn.phase:
                self.assertTrue(torch.isfinite(phase.grad).all())
                self.assertGreater(phase.grad.abs().sum().item(),0)

    def test_contrast_margin_has_expected_zero_loss(self):
        scores=torch.tensor([[.8,.2],[.3,.7]])
        self.assertEqual(contrast_loss(scores,torch.tensor([0,1])).item(),0.)


if __name__ == '__main__':
    unittest.main()
