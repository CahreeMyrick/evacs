import numpy as np
from evacs.model import load_model, predict
from evacs.features import FeatureTensor

def test_predict_label_in_set(tmp_path):
    labels = ["ambulance", "firetruck", "traffic"]
    model_path = tmp_path / "dummy.json"
    model_path.write_text('{"kind":"dummy_linear"}')
    model = load_model(str(model_path), labels)

    X = np.random.randn(64, 10).astype(np.float32)
    pred = predict(model, FeatureTensor(X=X))
    assert pred.label in labels
    assert abs(sum(pred.probs.values()) - 1.0) < 1e-3