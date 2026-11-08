# Ratchet

**一个本地优先的自强化引擎：让任何大模型，在你手里越用越懂你。**

[English](./README.md) · [架构说明](./docs/architecture.md) · [快速上手](./docs/quickstart.md) · [概念入门](./docs/concepts.md) · [治理设计](./docs/governance.md) · [怎么验收](./docs/how_to_verify.md) · [传到GitHub](./docs/upload_to_github.md)

Ratchet 不去微调你的模型。它在模型外面套一个闭环——**记忆、反思、择优、重写**——让每一次交互都给下一次留下点什么。任何能调用的模型都能用，包括那些你永远拿不到权重的闭源 API；不需要 GPU，一台笔记本就跑得动。

> **棘轮（ratchet）**：只能单向转动的齿轮。能力往上卡一格，就不会再滑回去。

---

## 一句话权限模型

**机器可以增，人只能删。**

Ratchet 学到的所有东西都是本地 SQLite 里的纯文本，你随时能看全。机器写入的条目是**仅可删除**的——你不能改写它，只能删掉它。删除只会减少能力，所以可以毫无顾虑地开放；而新增会改变模型在以后每一个任务里被告知什么，所以不能开放。

```bash
ratchet audit     # 看它自己得出了什么结论
ratchet rm af3c   # 删掉任何一条
ratchet panel     # 或者在浏览器里做这两件事
```

人往库里写东西的唯一通道，是**结构化、有界的纠正**：你挑一个维度（`too_long`、`wrong_tone`、`factually_wrong`……），再加一条上限 140 字符的附注；这条附注会先转义，再以「你的原话」身份被引用。系统里没有自由文本框——因为经验库会被注入以后每一个提示词，一个 textarea 等于开了第二条不受审计的指令通道。

---

## 快速上手

```bash
pip install ratchet-ai

ratchet init
ratchet add "回答用中文，三句话以内。"
ratchet chat "解释一下 LoRA" --task "解释概念"
ratchet audit
```

不需要 API key：默认后端是离线 echo，整个闭环照样跑得通，你可以先看清它到底在做什么，再决定花不花钱。要用真模型：

```bash
export OPENAI_API_KEY=sk-...
ratchet chat "解释一下 LoRA" --provider openai --model gpt-4o-mini
```

任何 OpenAI 兼容端点都行，包括 Ollama、vLLM、LM Studio 和千问 DashScope：

```bash
ratchet chat "你好" --provider openai --base-url http://localhost:11434/v1 --model qwen2.5
```

当库用：

```python
import ratchet
from ratchet.core.provider import OpenAICompatProvider

engine = ratchet.Engine(
    db_path="ratchet.db",
    provider=OpenAICompatProvider(model="gpt-4o-mini", api_key="..."),
)

result = engine.run(task="总结", user_input="本季度报告")
print(result.output)

engine.panel.view()          # 它学到了什么？
engine.panel.delete("af3c")  # 删掉不对的
```

---

## 它到底在做什么

一次 `run()` 走完整个循环：

```
检索     BM25 + 向量混合检索，召回相关经验
择优     汤普森采样挑一个策略——并且真的把它写进提示词
组装     系统提示 + 策略 + 画像 + 召回经验 + 你的请求，按这个顺序
生成     调用模型
验证     给输出打分：你的判定 > 客观校验 > 大模型裁判
记录     写审计轨迹；更新该策略的后验分布
反思     分低的时候，蒸馏出一条规则留给下次
学习     把这次交互汇入对你的推断画像
自动化   周期性提出更好的策略、重新压缩已学到的东西
```

后三步就是复利的来源：第 N 次交互写下的东西，第 N+1 次能取用。

### 这是强化学习，只是没有权重

| 强化学习要素 | Ratchet 里的对应 |
|---|---|
| 状态 State | 当前任务 + 你的推断画像 + 为它召回的经验 |
| 动作 Action | 用哪个策略回答（`concise`、`chain_of_thought`，以及它自己进化出来的） |
| 奖励 Reward | 验证器得分，按可信度分级：你的判定 > 客观校验 > 大模型裁判 |
| 策略 Policy | 每个策略一个 Beta 后验，由汤普森采样维护 |
| 策略更新 | `alpha += reward`，`beta += 1 - reward` |

策略的形式从神经网络换成了一组概率分布加可读文本规则，但更新的数学是一样的。这就是学界说的 **上下文强化学习（In-Context RL）**——Reflexion、Voyager 和一系列 "LLM as optimizer" 的工作都在这个方向上，是真实发表过的思路，不是我们打的比方。

奖励信号有两处刻意的强硬，因为一个设错的奖励比没有奖励更糟：

- **你的点踩是覆盖，不是参与混合。** 把一次否定和「客观校验通过」平均一下大约是 0.55，高于中点——那会**提高**你刚刚否掉的那个策略被选中的概率。
- **大模型裁判只是兜底**：权重封顶、带长度惩罚，且永远排在任何可机械校验的信号后面。

### 两条路线，同一份数据

- **路线 B（默认）**：不动权重。对所有模型都有效，包括闭源 API。上面的闭环就是它。
- **路线 A（可选）**：真训练。路线 B 积累的一切，本身就是一份从你真实工作里长出来的偏好数据集；导出它，在开源模型上跑 LoRA + DPO。

```bash
ratchet export --format dpo --out pairs.jsonl
ratchet train --dataset pairs.jsonl --base-model Qwen/Qwen2.5-7B --out-dir ./adapter
```

闭源模型走路线 A 在物理上不可能——你没法更新看不见的权重——而且它需要 GPU，所以它放在 `pip install 'ratchet-ai[train]'` 后面，核心包永远不会 import 它。

---

## 治理面板

```bash
ratchet panel
```

在 `127.0.0.1` 打开一个本地面板，列出每一条经验、它的来源、以及它实际贡献了什么——还有对你的推断画像，以及每条判断背后的证据。删除和恢复都是一次点击。没有自由文本编辑框。

```bash
ratchet audit --provenance auto    # 只看机器写的
ratchet explain af3c               # 这条为什么存在？
ratchet restore af3c               # 撤销一次删除
ratchet purge af3c                 # 物理清除（会先确认）
ratchet profile                    # 它自以为了解了你什么？
ratchet profile --reset            # 全部忘掉
ratchet correct --kind too_long --scope global --note "三点以内"
```

---

## 推断式用户画像

Ratchet 会自己琢磨你喜欢被怎么回答，不需要你手写设定。它用的信号都是可观测的事实——你把某个回答标成了「太长」、你的输入主要是中文、你接受了的某个回答很短——而不是模型对你的猜测，所以每一条判断都能追溯到产生它的那些交互。

三件事让它不容易悄悄跑偏：

- **置信度会衰减。** 长时间没有新证据支撑的判断，权威会慢慢下降。没有这条，「这个用户喜欢简短回答」就会自我确认：回答得简短 → 你没纠正 → 它把沉默当成认同。
- **反面证据能推翻它。** 支持和反驳分开计数；反面证据够了就翻转取值，而不是被平均进去。
- **你的话优先。** `ratchet profile --set verbosity=detailed` 设下的判断，机器永远不会推翻。

---

## 边界（说清楚的部分）

诚实写出来，比夸大宣传更能赢得信任。

- **它教不会模型本来不会的东西。** 不会写 Rust 的模型，装了还是不会。提升的是「调用已有能力的效率」和「不再重复犯错」。
- **路线 B 不动权重。** 数据库没了，学到的也就没了——这也正是它可逆的原因。
- **底座越弱，收益越小。** 反思本身需要模型有能力对自己的失败做推理。
- **收益会边际递减。** 库越大，检索精度越差。蒸馏和低分自动下架能延缓，但消不掉。
- **路线 A 不保证有提升。** 微调可能无效甚至有害，还可能让你丢掉从没打算舍弃的能力。训到独立的 adapter 目录，别动底座，上之前用 `ratchet eval` 量一下。
- **「任何大模型」指的是任何有文本接口的模型**，不包括无法编程调用的封闭产品。

---

## 安装与开发

```bash
pip install -e ".[dev]"
pytest
```

核心包**零第三方运行时依赖**，整个测试套件离线可跑、不需要 API key，所以 CI 又快又免费。

```bash
pip install 'ratchet-ai[train]'   # 路线 A：trl / peft / datasets
pip install 'ratchet-ai[mcp]'     # MCP 服务，接 Cursor、Claude Desktop 等
```

---

## 路线图

| 版本 | 主题 | 状态 |
|---|---|---|
| v0.1 | 记忆闭环：存储、检索、验证器、反思器、多臂老虎机 | 已完成 |
| v0.2 | 选择真正影响提示词；推断画像；提示进化；蒸馏；治理面板；评测；路线 A 导出 | **当前** |
| v0.3 | 内置更丰富的客观验证器（跑测试、校验 schema、检查编译） | 计划中 |
| v0.4 | 在 `Embedder` 协议后接入真实嵌入模型；更多 agent 适配器 | 计划中 |

---

## 许可证

MIT。见 [LICENSE](./LICENSE)。
