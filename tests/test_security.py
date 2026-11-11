from src.security import APIKeyAuth, generate_api_key


def test_api_key_auth_accepts_correct_key():
    auth = APIKeyAuth("secret")
    assert auth.verify("secret") is True


def test_api_key_auth_rejects_wrong_key():
    auth = APIKeyAuth("secret")
    assert auth.verify("wrong") is False
    assert auth.verify(None) is False
    assert auth.verify("") is False


def test_extract_from_bearer_header():
    auth = APIKeyAuth("x")
    assert auth.extract_from_header("Bearer abc") == "abc"
    assert auth.extract_from_header("abc") == "abc"
    assert auth.extract_from_header(None) is None


def test_generate_api_key_length():
    key = generate_api_key(48)
    assert isinstance(key, str)
    assert len(key) >= 48


def test_empty_key_raises():
    import pytest
    with pytest.raises(ValueError):
        APIKeyAuth("")
