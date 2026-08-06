import numpy as np
import pytest

from II_analyses.static_encoding import compute_participation_ratio


def test_compute_participation_ratio_for_equal_variance_axes():
    # Centered orthogonal axes with equal variance occupy two dimensions.
    space = np.array(
        [
            [1.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -1.0],
        ]
    )

    participation_ratio = compute_participation_ratio(space)

    assert participation_ratio == pytest.approx(2.0)
# EOF


def test_compute_participation_ratio_for_constant_space():
    space = np.ones((3, 5))

    participation_ratio = compute_participation_ratio(space)

    assert participation_ratio == 0.0
# EOF
