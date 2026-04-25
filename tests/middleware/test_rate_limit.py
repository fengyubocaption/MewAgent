"""限流模块单元测试。"""
import pytest
from unittest.mock import MagicMock
from backend.middleware.rate_limit import get_client_ip


class TestGetClientIp:
    """测试 get_client_ip 函数。"""

    def test_x_forwarded_for_first_ip(self):
        """X-Forwarded-For 存在时返回第一个 IP。"""
        request = MagicMock()
        request.headers = {"X-Forwarded-For": "192.168.1.1, 10.0.0.1"}
        request.client = MagicMock(host="127.0.0.1")

        result = get_client_ip(request)
        assert result == "192.168.1.1"

    def test_x_real_ip(self):
        """X-Real-IP 存在时返回该 IP。"""
        request = MagicMock()
        request.headers = {"X-Real-IP": "203.0.113.1"}
        request.client = MagicMock(host="127.0.0.1")

        result = get_client_ip(request)
        assert result == "203.0.113.1"

    def test_fallback_to_client_host(self):
        """没有代理头时回退到 client.host。"""
        request = MagicMock()
        request.headers = {}
        request.client = MagicMock(host="127.0.0.1")

        result = get_client_ip(request)
        assert result == "127.0.0.1"

    def test_no_client_returns_unknown(self):
        """没有 client 对象时返回 unknown。"""
        request = MagicMock()
        request.headers = {}
        request.client = None

        result = get_client_ip(request)
        assert result == "unknown"
