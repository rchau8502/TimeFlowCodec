import numpy as np
import pytest

from timeflowcodec.bjontegaard import bd_psnr, bd_rate


def test_bjontegaard_identical_curves():
    rates = [100.0, 200.0, 400.0, 800.0]
    psnrs = [30.0, 35.0, 39.0, 42.0]
    
    rate_diff = bd_rate(rates, psnrs, rates, psnrs)
    psnr_diff = bd_psnr(rates, psnrs, rates, psnrs)
    
    assert abs(rate_diff) < 1e-4
    assert abs(psnr_diff) < 1e-4


def test_bjontegaard_better_codec():
    # Anchor codec
    r_anchor = [200.0, 400.0, 800.0, 1600.0]
    d_anchor = [32.0, 36.0, 40.0, 44.0]
    
    # Better codec achieves identical quality at ~50% the bitrate
    r_test = [100.0, 200.0, 400.0, 800.0]
    d_test = [32.0, 36.0, 40.0, 44.0]
    
    rate_saving = bd_rate(r_anchor, d_anchor, r_test, d_test)
    psnr_gain = bd_psnr(r_anchor, d_anchor, r_test, d_test)
    
    # Should show ~ -50% BD-rate (bitrate reduction)
    assert -55.0 < rate_saving < -45.0
    # Should show positive BD-PSNR gain
    assert psnr_gain > 2.0


def test_bjontegaard_validation_errors():
    with pytest.raises(ValueError, match="at least 4"):
        bd_rate([100, 200], [30, 35], [100, 200], [30, 35])
        
    with pytest.raises(ValueError, match="at least 4"):
        bd_psnr([100, 200], [30, 35], [100, 200], [30, 35])


def test_bjontegaard_infinite_psnr():
    r1 = [100.0, 200.0, 400.0, 800.0]
    d1 = [35.0, 40.0, 45.0, 50.0]

    r2 = [50.0, 100.0, 200.0, 400.0]
    # Includes exact lossless / infinite PSNR
    d2 = [40.0, 50.0, 80.0, float("inf")]

    rate_saving = bd_rate(r1, d1, r2, d2)
    psnr_gain = bd_psnr(r1, d1, r2, d2)

    assert isinstance(rate_saving, float)
    assert isinstance(psnr_gain, float)
    assert not np.isnan(rate_saving)
    assert not np.isnan(psnr_gain)
    assert rate_saving < 0.0
    assert psnr_gain > 0.0
