import numpy as np

import libpysal
import spreg


def _build_inputs():
    db = libpysal.io.open(libpysal.examples.get_path("columbus.dbf"), "r")
    y = np.asarray(db.by_col("CRIME")).reshape(-1, 1)
    x = np.column_stack([db.by_col("INC"), db.by_col("HOVAL")])
    w = libpysal.weights.Rook.from_shapefile(
        libpysal.examples.get_path("columbus.shp")
    )
    w.transform = "r"
    return y, x, w


def test_stge_classic_forwards_white_covariance_to_selected_lag_model():
    y, x, w = _build_inputs()
    result, observed = spreg.stge_classic(
        y,
        x,
        w,
        robust="white",
        sig2n_k=True,
        p_value=0.01,
        w_lags=2,
        mprint=False,
        name_y="CRIME",
        name_x=["INC", "HOVAL"],
    )
    reference = spreg.GM_Lag(
        y,
        x,
        w=w,
        robust="white",
        w_lags=2,
        hard_bound=True,
        name_y="CRIME",
        name_x=["INC", "HOVAL"],
    )

    assert result == "LAG"
    assert observed.robust == "white"
    np.testing.assert_allclose(observed.betas, reference.betas)
    np.testing.assert_allclose(observed.vm, reference.vm)


def test_stge_classic_unadjusted_control_is_unchanged():
    y, x, w = _build_inputs()
    _, observed = spreg.stge_classic(
        y,
        x,
        w,
        robust=None,
        sig2n_k=True,
        p_value=0.01,
        w_lags=2,
        mprint=False,
        name_y="CRIME",
        name_x=["INC", "HOVAL"],
    )
    reference = spreg.GM_Lag(
        y,
        x,
        w=w,
        robust=None,
        w_lags=2,
        hard_bound=True,
        name_y="CRIME",
        name_x=["INC", "HOVAL"],
    )
    np.testing.assert_allclose(observed.vm, reference.vm)
