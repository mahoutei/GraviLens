"""Physics gates: bounds, coordinate convention, differentiability."""

import pytest
import torch

from gravilens.physics import (
    PIXEL_SCALE,
    THETA_E_MAX,
    THETA_E_MIN,
    sis_image_positions,
    sis_invert_bilinear,
    sis_invert_scatter,
    theta_from_unit,
)


def test_theta_scaling_stays_in_simulator_bounds():
    raw = torch.tensor([-50.0, 0.0, 50.0])
    theta = theta_from_unit(raw, THETA_E_MIN, THETA_E_MAX)
    assert theta[0].item() == pytest.approx(THETA_E_MIN, abs=1e-6)
    assert theta[2].item() == pytest.approx(THETA_E_MAX, abs=1e-6)
    assert THETA_E_MIN < theta[1].item() < THETA_E_MAX


def test_zero_theta_inverse_returns_the_image():
    torch.manual_seed(0)
    image = torch.rand(2, 64, 64)
    source = sis_invert_bilinear(image, torch.zeros(2), PIXEL_SCALE)
    expected = image / image.amax(dim=(1, 2), keepdim=True)
    # Ignore the two border pixels where the inside-mask can round differently.
    assert torch.allclose(source[:, 0, 2:-2, 2:-2], expected[:, 2:-2, 2:-2], atol=1e-5)


def test_point_source_maps_back_to_its_source_position():
    size, theta_e = 64, 1.0
    half = (size - 1) / 2.0
    beta_x, beta_y = 2 * PIXEL_SCALE, -3 * PIXEL_SCALE
    x1, y1, x2, y2 = sis_image_positions(
        torch.tensor([beta_x]), torch.tensor([beta_y]), torch.tensor([theta_e]))

    def to_pixel(arcsec: torch.Tensor) -> int:
        return int(round(float(arcsec) / PIXEL_SCALE + half))

    image = torch.zeros(1, size, size)
    for x, y in ((x1, y1), (x2, y2)):
        image[0, to_pixel(y), to_pixel(x)] = 1.0

    source = sis_invert_bilinear(image, torch.tensor([theta_e]), PIXEL_SCALE)
    row, col = divmod(int(source[0, 0].argmax()), size)
    assert abs(col - to_pixel(torch.tensor(beta_x))) <= 2
    assert abs(row - to_pixel(torch.tensor(beta_y))) <= 2


def test_theta_gradient_reaches_the_inverse():
    torch.manual_seed(0)
    image = torch.rand(2, 64, 64)
    theta = torch.tensor([0.8, 1.2], requires_grad=True)
    source = sis_invert_bilinear(image, theta, PIXEL_SCALE)
    source.sum().backward()
    assert theta.grad is not None
    assert torch.isfinite(theta.grad).all()
    assert theta.grad.abs().sum() > 0


def test_scatter_reference_is_shape_correct_and_not_differentiable():
    torch.manual_seed(0)
    image = torch.rand(2, 64, 64)
    theta = torch.tensor([0.8, 1.2], requires_grad=True)
    out = sis_invert_scatter(image, theta, PIXEL_SCALE)
    assert out.shape == (2, 1, 64, 64)
    assert not out.requires_grad
