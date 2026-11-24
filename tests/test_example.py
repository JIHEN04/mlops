import model_pipeline as mp


def test_prepare_data_returns_non_empty_sets():
    x_train, x_test, y_train, y_test = mp.prepare_data("Churn_Modelling.csv")
    assert len(x_train) > 0
    assert len(x_test) > 0
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)


def test_train_model_has_predict():
    x_train, _, y_train, _ = mp.prepare_data("Churn_Modelling.csv")
    model = mp.train_model(x_train, y_train, n_estimators=10)
    assert hasattr(model, "predict")
