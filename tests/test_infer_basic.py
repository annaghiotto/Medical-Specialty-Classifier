import numpy as np
import pytest

from src.infer import MedicalSpecialtyPredictor


@pytest.fixture(scope="session")
def predictor():
    return MedicalSpecialtyPredictor("artifacts")

def test_encode_not_empty_and_normalized(predictor):
    texts = [
        "HPI: 67-year-old male with chest pain radiating to the left arm. ECG shows anterior ST elevation.",
        "A 34-year-old male with knee effusion; MRI confirms medial meniscus tear."
    ]
    X = predictor._encode(texts)
    assert X.ndim == 2 and X.shape[0] == 2 and X.shape[1] > 0
    norms = np.linalg.norm(X, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)

def test_predict_top1_and_top3_formats(predictor):
    text = "A 67-year-old male presents with chest pain radiating to the left arm and shortness of breath for two hours. ECG reveals ST-segment elevation in anterior leads and troponin is positive. Patient was started on aspirin, heparin, and taken for emergency coronary angiography showing LAD occlusion successfully stented."
    top1 = predictor.predict(text, topk=1)
    top3 = predictor.predict(text, topk=3)
    assert isinstance(top1, list) and len(top1) == 1 and isinstance(top1[0], str)
    assert isinstance(top3, list) and len(top3) == 1 and isinstance(top3[0], list)
    assert len(top3[0]) == 3 and all(isinstance(lbl, str) for lbl in top3[0])
