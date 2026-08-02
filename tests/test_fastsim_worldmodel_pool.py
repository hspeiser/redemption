import torch

from aigp.fastsim.worldmodel import ResidualEnsemble, ResidualEnsemblePool


def _constant_ensemble(x_shift: float, y_shift: float) -> ResidualEnsemble:
    model = ResidualEnsemble(members=2, input_dim=17, hidden=8, output_dim=9)
    with torch.no_grad():
        model.x_mean.fill_(x_shift)
        model.x_std.fill_(2.0)
        model.y_mean.fill_(y_shift)
        model.y_std.fill_(1.0)
        for parameter in model.parameters():
            parameter.zero_()
    return model


def test_pool_preserves_per_checkpoint_normalization_and_member_order():
    first = _constant_ensemble(0.0, 1.0)
    second = _constant_ensemble(4.0, 2.0)
    pool = ResidualEnsemblePool([first, second])
    features = torch.zeros(3, 17)

    means, log_stds = pool(features)
    support = pool.support_z_by_member(features)

    assert pool.member_count == 4
    assert means.shape == (4, 3, 9)
    assert log_stds.shape == (4, 3, 9)
    assert torch.allclose(means[:2], torch.ones_like(means[:2]))
    assert torch.allclose(means[2:], torch.full_like(means[2:], 2.0))
    assert torch.allclose(support[:2], torch.zeros_like(support[:2]))
    assert torch.allclose(support[2:], torch.full_like(support[2:], 2.0))
