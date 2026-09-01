import pytest

from src.api_client import APIError


def test_api_error_is_exception():
    with pytest.raises(APIError):
        raise APIError("example")

