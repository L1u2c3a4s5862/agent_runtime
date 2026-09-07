import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from loguru import logger

from httpx import Client
from openai import (
    OpenAI, APIConnectionError,
    APIError, APIStatusError
)
from openai.types.chat import ChatCompletion

from .context import Message
from .errors import LLMError

load_dotenv()

class LLMClient(ABC):
    """LLM 抽象接口。"""

    @abstractmethod
    def complete(self, messages: list[Message]) -> str:
        """让模型补全一轮对话。"""

class OpenAIClient(LLMClient):
    """OpenAI 兼容 /chat/completions 客户端，底层使用 openai SDK。"""

    def __init__(
        self,
        base_url: str='',
        api_key: str='',
        model: str='',
        timeout_s: float=30.0,
        temperature: float=0.0,
        http_client: Optional[Client]=None
    ):
        self.base_url = base_url or os.getenv('LLM_BASE_URL', '')
        self.api_key = api_key or os.getenv('LLM_API_KEY', '')
        self.model = model or os.getenv('LLM_MODEL', '')
        if not self.base_url:
            raise ValueError('base_url 未配置（可通过参数或 LLM_BASE_URL 环境变量提供）')
        if not self.model:
            raise ValueError('model 未配置（可通过参数或 LLM_MODEL 环境变量提供）')
        self.temperature = temperature
        self._client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=timeout_s,
            http_client=http_client
        )

    def close(self):
        """释放底层 HTTP 连接。"""
        self._client.close()

    def complete(self, messages: list[Message]) -> str:
        """调用 chat/completions 并提取回复文本。"""
        response = self._create(messages)
        content = self._extract_content(response)
        if not content:
            # 空输出排查依赖原始响应细节（finish_reason、reasoning_content 等）
            logger.debug(f'LLM 空输出，原始响应: {str(response.model_dump())[:1000]!r}')
            raise LLMError('LLM 返回空输出')
        return content

    def _create(self, messages: list[Message]) -> ChatCompletion:
        """通过 openai SDK 发送 chat/completions 请求。"""
        started = datetime.now()
        logger.debug(
            f'LLM 请求: model={self.model!r} messages={len(messages)} '
            f'temperature={self.temperature} max_tokens=512'
        )
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[message.to_openai_dict() for message in messages],
                temperature=self.temperature,
                max_tokens=512
            )
        except APIStatusError as e:
            raise LLMError(f'LLM API 返回错误 HTTP {e.status_code}: {e.message[:200]}') from e
        except APIConnectionError as e:
            raise LLMError(f'LLM API 调用失败: {e}') from e
        except APIError as e:
            raise LLMError(self._describe_api_error(e)) from e
        # 非 JSON content-type 时 SDK 跳过解析，原样返回 body 文本
        if not isinstance(response, ChatCompletion):
            raise LLMError(f'LLM API 响应不是合法 JSON: {response!r}')
        logger.debug(f'LLM 响应: 耗时={(datetime.now() - started).total_seconds():.2f}s')
        return response

    @staticmethod
    def _describe_api_error(error: APIError) -> str:
        """把非状态码、非网络类的 APIError 翻译成可读消息。"""
        if isinstance(error.__cause__, json.JSONDecodeError):
            return f'LLM API 响应不是合法 JSON: {error.__cause__}'
        return f'LLM API 错误: {error}'

    @staticmethod
    def _extract_content(response: ChatCompletion) -> str:
        """从响应体提取 `choices[0].message.content`；推理模型 content 为空时兜底取 `reasoning_content`。"""
        if not response.choices:
            return ''
        message = response.choices[0].message
        content = message.content or getattr(message, 'reasoning_content', '') or ''
        return content if isinstance(content, str) else ''
