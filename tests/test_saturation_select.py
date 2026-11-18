

def test_bisect_mode_is_sublinear_in_measurements():
    # QC on review-batch: the dense default is free on precomputed grids
    # (the only production caller) but live-measurement callers need the
    # documented O(log n) contract back — scan="bisect"/"auto".
    from prismaquant.saturation_select import find_saturation_bpp

    calls = []

    def measure(bpp):
        calls.append(bpp)
        return (0.5 if bpp < 6.0 else 0.1, 0.01)

    grid = [4.0 + 0.25 * i for i in range(17)]  # 17 points
    out = find_saturation_bpp(grid, measure, z=2.0, scan="bisect")
    assert out["bpp"] == 6.0
    assert len(calls) <= 7, f"bisect must be O(log n), measured {len(calls)}"


def test_saturation_falls_back_to_the_highest_rate_when_no_band_accepts():
    import math

    from prismaquant.saturation_select import find_saturation_bpp

    values = {4.0: (0.5, 1.0), 6.0: (0.25, 1.0), 8.0: (0.125, 1.0)}
    result = find_saturation_bpp(list(values), values.__getitem__, z=-1.0)
    assert result == {
        "bpp": 8.0,
        "asymptote_bpp": 8.0,
        "kl_at_bstar": 0.125,
        "kl_asymptote": 0.125,
        "z": -1.0,
        "n_measurements": 3,
        "measured": {rate: {"kl": kl, "se": se} for rate, (kl, se) in values.items()},
        "slopes": [
            {"from": 4.0, "to": 6.0, "dKL_per_bit": -0.125},
            {"from": 6.0, "to": 8.0, "dKL_per_bit": -0.0625},
        ],
        "trace": [
            {"bpp": rate, "kl": kl, "se": se, "kl_asymptote": 0.125,
             "band": -math.sqrt(2.0), "within_noise": False}
            for rate, (kl, se) in values.items()
        ],
    }
