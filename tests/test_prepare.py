import pandera.errors
import pytest

from demandcast.data.schemas import validate_sales


def test_valid_frame_passes(tiny_sales):
    validated = validate_sales(tiny_sales)
    assert len(validated) == len(tiny_sales)


def test_negative_demand_rejected(tiny_sales):
    bad = tiny_sales.copy()
    bad.loc[bad.index[0], "y"] = -1.0
    with pytest.raises(pandera.errors.SchemaErrors):
        validate_sales(bad)


def test_duplicate_series_day_rejected(tiny_sales):
    bad = tiny_sales.copy()
    bad.loc[bad.index[1], ["unique_id", "ds"]] = bad.loc[bad.index[0], ["unique_id", "ds"]].values
    with pytest.raises(pandera.errors.SchemaErrors):
        validate_sales(bad)


def test_synthetic_generator_produces_valid_frame():
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "make_synthetic.py"
    spec = importlib.util.spec_from_file_location("make_synthetic", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    df = mod.generate(items=3, stores=1, days=60)
    validated = validate_sales(df)
    assert validated["unique_id"].nunique() == 3
    assert (validated["y"] >= 0).all()
