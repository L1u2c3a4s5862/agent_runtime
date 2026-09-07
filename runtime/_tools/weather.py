import os
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from loguru import logger

from httpx import Client

from ..errors import ToolError
from ..tools import Tool

load_dotenv()

_GEO_HOST = 'geoapi.qweather.com'

def make_weather_tool(http_client: Optional[Client]=None) -> Tool:
    """构造基于 QWeather 的 weather 工具。"""
    key = os.getenv('QWEATHER_API_KEY', '')
    host = os.getenv('QWEATHER_API_HOST', 'api.qweather.com')
    http = http_client or Client(timeout=10.0)
    return Tool(
        name='weather',
        description='查询城市天气：不给日期查实时天气，给日期（3 天内）查当天预报',
        parameters={
            'type': 'object',
            'properties': {
                'city': {'type': 'string', 'description': '城市名，如 北京'},
                'date': {'type': 'string', 'description': '日期，格式 YYYY-MM-DD；缺省返回实时天气'}
            },
            'required': ['city']
        },
        func=lambda city, date='': _weather(city, date, key, host, http)
    )

def _weather(city: str, date: str, key: str, host: str, http: Client) -> str:
    """查询实时天气或 3 天内预报。"""
    if not key:
        raise ToolError('未配置 QWEATHER_API_KEY（可在 https://dev.qweather.com 免费申请）')
    logger.debug(f'QWeather 查询: city={city!r} date={date or "实时"!r}')
    location = _resolve_location(city, key, http)
    logger.debug(f'QWeather 城市解析: {city!r} -> id={location["id"]} {_full_name(location)!r}')
    if not date:
        data = _get_qweather('/v7/weather/now', {'location': location['id']}, key, host, http)
        return _format_now(data, location)
    data = _get_qweather('/v7/weather/3d', {'location': location['id']}, key, host, http)
    return _format_forecast(data, location, date)

def _get_qweather(path: str, params: dict[str, str], key: str, host: str, http: Client) -> dict:
    """请求 QWeather API 并校验业务 code。"""
    started = datetime.now()
    response = http.get(f'https://{host}{path}', params={**params, 'key': key})
    response.raise_for_status()
    data = response.json()
    logger.debug(
        f'QWeather 请求: {path!r} code={data.get("code")} '
        f'耗时={(datetime.now() - started).total_seconds():.2f}s'
    )
    if data.get('code') != '200':
        raise ToolError(f'QWeather API 错误 code={data.get("code")}（key 无效或订阅额度不足）')
    return data

def _resolve_location(city: str, key: str, http: Client) -> dict:
    """城市名 → QWeather location 字典。"""
    data = _get_qweather('/v2/city/lookup', {'location': city}, key, _GEO_HOST, http)
    locations = data.get('location') or []
    if not locations:
        raise ToolError(f'未找到城市「{city!r}」')
    return locations[0]

def _full_name(location: dict) -> str:
    """省/市拼接全名，相邻重复段去重，如 北京市·朝阳。"""
    parts: list[str] = []
    for part in (location.get('adm1', ''), location.get('adm2', ''), location.get('name', '')):
        if part and (not parts or part != parts[-1]):
            parts.append(part)
    return '·'.join(parts)

def _format_now(data: dict, location: dict) -> str:
    """格式化实时天气。"""
    now = data['now']
    return (
        f'{_full_name(location)}实时天气（{now["obsTime"]} 观测）：{now["text"]}，'
        f'{now["temp"]}℃，体感 {now["feelsLike"]}℃，'
        f'{now["windDir"]} {now["windScale"]} 级，相对湿度 {now["humidity"]}%'
    )

def _format_forecast(data: dict, location: dict, date: str) -> str:
    """从 3 天预报中挑出指定日期。"""
    daily = data.get('daily') or []
    for day in daily:
        if day['fxDate'] == date:
            return (
                f'{_full_name(location)} {date}：白天 {day["textDay"]}，夜间 {day["textNight"]}，'
                f'{day["tempMin"]}~{day["tempMax"]}℃，{day["windDirDay"]} {day["windScaleDay"]} 级'
            )
    days = '、'.join(day['fxDate'] for day in daily)
    raise ToolError(f'仅支持 {days} 的预报（免费订阅 3 天），{date} 超出范围')
