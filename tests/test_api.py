from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from clickbait_classifier.api import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "OK", "status-code": 200}

@patch("src.clickbait_classifier.api.torch.load")
@patch("src.clickbait_classifier.api.os.path.getmtime")
@patch("src.clickbait_classifier.api.glob.glob")
@patch("src.clickbait_classifier.api.ClickbaitClassifier")
@patch("src.clickbait_classifier.api.AutoTokenizer")
def test_predict_endpoint(mock_tokenizer, mock_classifier, mock_glob, mock_mtime, mock_torch_load):
    # 1. Simuler filsystem-sjekkene i api.py
    mock_mtime.return_value = 123456789.0
    mock_glob.return_value = ["models/fake_model.ckpt"]

    # 2. Simuler at torch.load returnerer en state_dict (api.py linje 41)
    mock_torch_load.return_value = {"state_dict": {"model.weight": [0.1]}}

    # 3. Lag en "ordentlig" mock-modell som har .load_state_dict() og .eval()
    mock_model_obj = MagicMock()
    mock_classifier.return_value = mock_model_obj

    # 4. Simuler at selve prediksjonen i /predict fungerer
    # Dette sørger for at torch.argmax i api.py får noe å jobbe med
    mock_model_obj.return_value = MagicMock()

    with TestClient(app) as client:
        # Nå kjører startup_event helt uten fil-tilgang!
        response = client.post("/predict?text=Dette er en test")
        assert response.status_code == 200
        assert "is_clickbait" in response.json()
