from src.models import LinearRegression


def test_prediction(tmp_path):
    model = LinearRegression()
    data = tmp_path / "data.csv"

    data.write_text("km,price\n1000,50000\n2000,70000\n3000,90000\n")

    model.train(data, epoch=100000, learning_rate=0.0001)
    prediction = model.estimatePrice(1000)

    assert abs(50000 - prediction) <= 1e-6
