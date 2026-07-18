import json
import pathlib
import subprocess
import sys

from unittest.mock import patch
from config import MAIN
from src.models import LinearRegression


def test_load_valid_data(tmp_path):
    model = LinearRegression()
    path: pathlib.Path = tmp_path / "data.csv"

    path.write_text("x,y\n1,2\n3,4")

    df = model._loadData(path)

    assert df is not None
    assert len(df) == 2
    assert list(df.columns) == ["x", "y"]


def test_load_data_with_invalid_rows(tmp_path):
    model = LinearRegression()
    path: pathlib.Path = tmp_path / "data.csv"

    path.write_text("x,y\n1,2\nbad,2\n5,nope\n3,4\n")

    df = model._loadData(path)

    assert df is not None
    assert len(df) == 2
    assert list(df.columns) == ["x", "y"]
    assert df.iloc[0]["x"] == 1
    assert df.iloc[1]["x"] == 3


def test_load_data_with_duplicates(tmp_path):
    model = LinearRegression()
    path: pathlib.Path = tmp_path / "data.csv"

    path.write_text("x,y\n1,2\n1,2\n5,3\n3,4\n5,3")

    df = model._loadData(path)

    assert df is not None
    assert len(df) == 3
    assert list(df.columns) == ["x", "y"]
    assert df.iloc[0]["x"] == 1
    assert df.iloc[1]["x"] == 5
    assert df.iloc[2]["x"] == 3


def test_load_json_invalid(tmp_path):
    path = tmp_path / "model.json"

    data = {
        "theta0": "asd",
        "theta1": 1,
        "min0": 9,
        "max0": "abcx",
    }
    path.write_text(json.dumps(data))

    model = LinearRegression(path)

    assert model.theta0 == 0.0
    assert model.theta1 == 1
    assert model.min0 == 9
    assert model.max0 is None


def test_load_json_missing_data(tmp_path):
    path = tmp_path / "model.json"

    data = {
        "theta0": 2,
        "min0": 9,
        "max0": 1,
    }
    path.write_text(json.dumps(data))

    model = LinearRegression(path)

    assert model.theta0 == 2
    assert model.theta1 == 0.0
    assert model.min0 == 9
    assert model.max0 == 1


def test_load_json_no_minmax(tmp_path):
    path = tmp_path / "model.json"

    data = {
        "theta0": 2,
        "theta1": 2,
    }
    path.write_text(json.dumps(data))

    model = LinearRegression(path)

    assert model.theta0 == 2
    assert model.theta1 == 2
    assert model.min0 is None
    assert model.max0 is None


def test_load_json_num_too_big(tmp_path):
    path = tmp_path / "model.json"

    data = {
        "theta0": 2,
        "theta1": 10**1000,
    }
    path.write_text(json.dumps(data))

    model = LinearRegression(path)

    print(model.theta1)
    assert model.theta0 == 2
    assert model.theta1 == 0.0


def test_cli_success(tmp_path):
    data = tmp_path / "data.csv"

    data.write_text("km,price\n1000,50000\n2000,70000\n3000,90000\n")

    result = subprocess.run(
        [sys.executable, str(MAIN)],
        input=f"train {data}\nestimate 1500\n",
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "estimated price" in result.stdout.lower()
    assert "estimated price: 0" not in result.stdout.lower()


def test_cli_unknown_command(tmp_path):
    result = subprocess.run(
        [sys.executable, str(MAIN)],
        input="trai n\n",
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "unknown" in result.stdout.lower()


def test_cli_no_args(tmp_path):
    result = subprocess.run(
        [sys.executable, str(MAIN)],
        input="train\n",
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "usage" in result.stdout.lower()


def test_cli_rejects_invalid_number(tmp_path):
    result = subprocess.run(
        [sys.executable, str(MAIN)],
        input="estimate abs5\n",
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "error" in result.stdout.lower()
    assert "estimated price" not in result.stdout.lower()


def test_cli_rejects_invalid_file(tmp_path):
    result = subprocess.run(
        [sys.executable, str(MAIN)],
        input="train ./does_not_exist.csv",
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "error" in result.stdout.lower()

    result = None
    path = tmp_path / "no_permissions.csv"
    path.touch()

    real_open = open

    def fake_open(path, *args, **kwargs):
        if path == path:
            raise PermissionError
        return real_open(path, *args, **kwargs)

    with patch("builtins.open", side_effect=fake_open):
        result = subprocess.run(
            [sys.executable, str(MAIN)],
            input=f"graph {str(path)}",
            capture_output=True,
            text=True,
        )

    assert result.returncode == 0
    assert "error" in result.stdout.lower()
