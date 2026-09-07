# TROUBLESHOOT — AI 协作开发记录

---

## 1. Tavily 搜索 + QWeather 天气工具

**Prompt**：`请用Tavily开发搜索工具，Qweather开发天气查询工具`

**关键决策**（与用户确认）：
- **替换默认注册表**：`search`/`weather` 名字不变（Agent 无感），离线假数据版保留为 `make_offline_registry()` 供测试和离线演示
- Tavily 用官方 `tavily-python` SDK；QWeather 无官方 SDK，用 httpx 直连
- 依赖注入式测试：Tavily 注入 fake client，QWeather 注入 `httpx.MockTransport`，测试零 API 依赖
- key 缺失时**构造不报错、调用时才抛 `ToolError`**（Agent 能收到明确错误提示而非崩溃）

**处理**：新增 `web_tools.py`；QWeather 流程为 `城市名 → geo lookup → location id → 实时天气/3 天预报`，`date` 参数可选。

**遇到的问题与解决**：

| 问题 | 原因 | 解决 |
|---|---|---|
| `test_agent.py` 13 个测试打到真实 API | `make_agent` 默认用 `make_default_registry()`，换成真实工具后环境无 key 直接报错 | `make_agent` 默认挂离线假数据注册表（`kwargs.setdefault('tools', ...)`） |

**验证**：109 个测试全绿；Tavily 真实冒烟调用成功。

---

## 2. 重构 `_tools/` 包后的测试修复

**Prompt**：`pytest测试还差search和weather工具没有写测试代码, 请帮我写完pytest测试`

**背景**：用户把 `standard_tools.py` 重构拆分为 `_tools/` 包（calculator / search / weather 三模块），离线假数据工具被删除，测试 19 个断裂。

**处理**：
- 新建 `tests/test_search.py`（4 用例）、`tests/test_weather.py`（6 用例）
- 删除 `test_tools.py` 中 6 个测试已删除离线工具的过时用例（API 已对不上，留必红）
- 重建 `test_agent.py` 的 `make_agent` helper，内嵌假数据 search/weather 工具（calculator 用真实现）

**验证**：103 个测试全绿。

---

## 3. QWeather 真实冒烟：城市名重复显示

**Prompt**：`我设置了QWEATHER_API_KEY的系统参数，请帮我完成测试`

**处理**：确认环境变量设置不破坏测试隔离（全部测试用 monkeypatch + MockTransport），真实调用 QWeather API 验证。

**遇到的问题与解决**：

| 问题 | 原因 | 解决 |
|---|---|---|
| 输出「北京市·北京·北京」 | geo API 对「北京」返回的 `adm2`（市）与 `name`（区县）相同，`_full_name` 三段拼接未去重 | 相邻重复段去重；先写测试 `test_adjacent_duplicate_segments_deduped` 再改实现 |

**验证**：104 个测试全绿；真实输出修复为「北京市·北京实时天气」。

---

## 4. 用户输入与 Agent 回复写日志

**Prompt**：`请将我将我的问题和Agent的恢复也保存到日志 (DEBUG) 中方便排查`（原文「恢复」应为「回复」）

**处理**：`agent.py` 的 `run()` 中，用户输入与 Agent 最终回复各加一条 `logger.debug`，与既有「工具执行成功」「回合结束」日志串成完整链路。

**顺带报告（未修）**：`agent.py`/`parser.py`/`tools.py`/`app.py` 各自 `logger.add('agent.log')`，loguru 全局 logger 每 add 一次多写一遍，日志重复 3 份。

**验证**：104 个测试全绿；日志验证链路完整。

---

## 5. 全链路日志增强

**Prompt**：`所有调用日志的地方或还可以添加调用日志的地方，请将日志信息输出更详细`

**处理**（各层职责）：

- **agent.py**：所有日志带 `session id`；LLM 调用（模型/消息数/耗时/字数）、LLM 失败（ERROR）、解析失败、未知工具/参数非法、工具执行（参数/结果前 200 字/耗时）、工具异常（ERROR）、超步数中止（ERROR）
- **llm.py**：请求参数（model/messages/temperature/max_tokens）+ 响应耗时
- **search.py / weather.py**：query、结果数、城市解析、每次 API 请求的 path/code/耗时
- **app.py**：`logger.error` → `logger.exception`（带堆栈）

**插曲**：编辑 `weather.py` 时误删 `_resolve_location` 函数——恰好被刚加的日志抓了现行（`NameError`），补回后恢复全绿。**日志增强的直接收益**。

**验证**：104 个测试全绿；一个真实回合的日志可完整还原调用链。

---

## 6. 「LLM 返回空输出」排查

**Prompt**：`请查看aqgent.log日志文件，为什么会报错[ERROR] LLM 返回空输出`（原文「aqgent」为 agent 笔误）

**日志证据链**（日志增强后的直接收益）：
```
用户输入「这个网站中都有什么」→ 模型调 Tavily（两次各近 2000 字结果入上下文）
→ 第三次 LLM 调用：19 条消息、输入很大、max_tokens=512
→ API 正常返回（5.21s、无 HTTP 错误），但 choices[0].message.content 为空
→ 代码判空 → LLMError('LLM 返回空输出')
```

**根因分析**（两个可能）：
1. 推理模型把内容放进 `reasoning_content` 字段，`content` 为空
2. 长输入 + 小 max_tokens 下服务端返回空 completion

**解决**：
- `_extract_content` 兜底：content 为空时取 `reasoning_content`
- 空输出时把**原始响应**（finish_reason、reasoning_content 等）写进日志，下次复现一眼定位
- 新增 2 个测试（reasoning 兜底、content 优先）

**验证**：106 个测试全绿。

---

## 7. 会话持久化与恢复

**Prompt**：`怎么在app.py中进入上一个会话继续对话`

**现状分析**：`Session` 纯内存，历史存在 `ContextManager` 内存列表，进程退出即失，磁盘上没有「上一个会话」可进。

**关键决策**（与用户确认）：直接实现；进入方式选**命令行参数指定**。

**处理**：
- `Session.to_dict()`/`from_dict()`：序列化核心状态（历史/参数/metadata）；traces 为诊断数据不持久化
- `save_session()`/`load_session()`：JSON 文件 IO，路径参数化便于测试
- `app.py`：`-s/--session <id>` 恢复，缺省新建；每回合成功后自动落盘 `sessions/<id>.json`
- `.gitignore` 补 `sessions/`（对话内容含隐私）

**验证**：
- 110 个测试全绿（新增 4 个：dict 往返、文件往返、未知会话报错、恢复后继续对话再落盘）
- 端到端实测：第一轮问「用一句话自我介绍」→ 退出 → `-s` 恢复 → 问「我刚才问你的第一个问题是什么？」→ 模型准确答出上一轮提问

---

## 8. pytest 日志分离到 test.log

**Prompt**：`请将pytest中的日志保存到其他文件中如test.log`

**处理**：新建 `tests/conftest.py`，用 `pytest_collection_finish` 钩子做日志分流：

```python
def pytest_collection_finish(session):
    logger.remove()          # 摘掉各模块 import 时注册的 agent.log handler
    setup_log_file('test.log')
```

**原理**：pytest 先 import 全部测试模块（此时 runtime 各模块顶层已注册 `agent.log` handler）→ collection 完成 → 钩子换 handler → 测试日志全部进 `test.log`，**产品代码零改动**。

**验证**：110 全绿；`test.log` 29KB，`agent.log` 0B 未被污染。
