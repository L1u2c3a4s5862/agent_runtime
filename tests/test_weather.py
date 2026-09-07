from typing import Any

from httpx import Client, MockTransport, Request, Response
from pytest import MonkeyPatch, raises

from runtime._tools.weather import make_weather_tool
from runtime.errors import ToolError

_GEO_RESPONSE = {'code': '200', 'location': [
    {'name': '朝阳', 'id': '101010300', 'adm2': '北京', 'adm1': '北京市'}
]}

_NOW_RESPONSE = {'code': '200', 'now': {
    'obsTime': '2026-09-05T15:00+08:00', 'text': '多云', 'temp': '19',
    'feelsLike': '17', 'windDir': '东北风', 'windScale': '2', 'humidity': '45'
}}

_FORECAST_RESPONSE = {'code': '200', 'daily': [{
    'fxDate': '2026-09-05', 'textDay': '多云', 'textNight': '晴',
    'tempMax': '28', 'tempMin': '17', 'windDirDay': '东北风', 'windScaleDay': '2'
}, {
    'fxDate': '2026-09-06', 'textDay': '小雨', 'textNight': '阴',
    'tempMax': '25', 'tempMin': '18', 'windDirDay': '南风', 'windScaleDay': '3'
}]}

def make_client(path_responses: dict[str, dict], monkeypatch: MonkeyPatch) -> tuple[Any, dict[str, dict]]:
    """构造离线 QWeather 工具，返回 (tool, 请求参数记录)。"""
    monkeypatch.setenv('QWEATHER_API_KEY', 'test-key')
    captured: dict[str, dict] = {}

    def handler(request: Request) -> Response:
        captured[request.url.path] = dict(request.url.params)
        return Response(200, json=path_responses[request.url.path])

    tool = make_weather_tool(http_client=Client(transport=MockTransport(handler)))
    return tool, captured

class TestMakeWeatherTool:
    def test_adjacent_duplicate_segments_deduped(self, monkeypatch: MonkeyPatch):
        # geo 对「北京」返回的 adm2 与 name 相同，拼接时应去重
        geo = {
            'code': '200',
            'location': [{
                'name': '北京', 'id': '101010100',
                'adm2': '北京', 'adm1': '北京市'
            }]
        }
        tool, _ = make_client({'/v2/city/lookup': geo, '/v7/weather/now': _NOW_RESPONSE}, monkeypatch)
        output = tool.func('北京')
        assert '北京市·北京·北京' not in output
        assert '北京市·北京实时天气' in output

    def test_now_weather(self, monkeypatch: MonkeyPatch):
        tool, captured = make_client(
            {'/v2/city/lookup': _GEO_RESPONSE, '/v7/weather/now': _NOW_RESPONSE}, monkeypatch
        )
        output = tool.func('北京')
        assert '北京市·北京·朝阳' in output
        assert '多云' in output and '19℃' in output and '45%' in output
        # geo 用城市名，天气接口用解析出的 location id
        assert captured['/v2/city/lookup']['location'] == '北京'
        assert captured['/v7/weather/now']['location'] == '101010300'
        assert all(params['key'] == 'test-key' for params in captured.values())

    def test_forecast_by_date(self, monkeypatch: MonkeyPatch):
        tool, _ = make_client(
            {'/v2/city/lookup': _GEO_RESPONSE, '/v7/weather/3d': _FORECAST_RESPONSE}, monkeypatch
        )
        output = tool.func('北京', '2026-09-06')
        assert '2026-09-06' in output and '小雨' in output and '18~25℃' in output

    def test_date_out_of_range_raises(self, monkeypatch: MonkeyPatch):
        tool, _ = make_client(
            {'/v2/city/lookup': _GEO_RESPONSE, '/v7/weather/3d': _FORECAST_RESPONSE}, monkeypatch
        )
        with raises(ToolError, match='超出范围'):
            tool.func('北京', '2026-09-10')

    def test_city_not_found_raises(self, monkeypatch: MonkeyPatch):
        tool, _ = make_client({'/v2/city/lookup': {'code': '200', 'location': []}}, monkeypatch)
        with raises(ToolError, match='未找到城市'):
            tool.func('不存在市')

    def test_missing_key_raises(self, monkeypatch: MonkeyPatch):
        monkeypatch.delenv('QWEATHER_API_KEY', raising=False)
        tool = make_weather_tool()
        with raises(ToolError, match='QWEATHER_API_KEY'):
            tool.func('北京')

    def test_api_error_code_raises(self, monkeypatch: MonkeyPatch):
        tool, _ = make_client(
            {'/v2/city/lookup': _GEO_RESPONSE, '/v7/weather/now': {'code': '401'}}, monkeypatch
        )
        with raises(ToolError, match='code=401'):
            tool.func('北京')
