# 怎么验收这个项目（照着敲就行）

> 你只需要会一件事：**打开终端，把下面的命令粘进去，回车。**
> 全程约 6 分钟。不需要联网、不需要显卡、不需要 API key。
> 如果哪一步看不懂，直接跳到第 6 节。

---

## 第 0 步：打开终端并进到项目文件夹

**Windows**：打开 `ratchet` 文件夹，点一下顶部的地址栏，输入 `cmd` 然后回车。会弹出一个黑窗口。

**Mac**：右键 `ratchet` 文件夹 →「服务」→「新建位于文件夹位置的终端」。

黑窗口里应该显示你正处在 `ratchet` 这个文件夹里。如果不是，输入：

```bash
cd ratchet
```

---

## 第 1 步：装它（10 秒）

```bash
pip install -e .
```

**如果报错说找不到 pip，或者网络不通**，改用下面这行代替（每次新开窗口都要敲一次）：

```bash
set PYTHONPATH=src          # Mac/Linux 用：export PYTHONPATH=src
```

---

## 第 2 步：第一条命令（3 秒）

```bash
python scripts/verify.py
```

**看到这样就成功了：**

```
Ratchet acceptance checks
============================================================
  [PASS] strategy reaches prompt
  [PASS] delimiter escape
  [PASS] machine entries delete only
  [PASS] correction is escaped
  [PASS] profile decays
  [PASS] owner outranks inference
  [PASS] store is thread safe
  [PASS] secret redaction
  [PASS] packaging
============================================================
9/9 checks passed
```

最后一行是 `9/9 checks passed`。**只要不是 9/9，就是有问题，把屏幕上的内容发给我。**

这 9 项分别在说什么，一句话版：

| 检查 | 一句话 |
|---|---|
| strategy reaches prompt | 系统选的说话方式，真的传给了 AI（这是最要命的一个，原来它是断的） |
| delimiter escape | 学到的话里不能藏着能骗 AI 的假指令 |
| machine entries delete only | AI 自己学的东西**你只能删不能改** |
| correction is escaped | 你纠正它时写的备注，会被当成"你的反应"而不是"你的命令" |
| profile decays | 很久不用的推断会自动失效，不会一条错一辈子 |
| owner outranks inference | 你手动设的东西，AI 的推断永远推翻不了 |
| store is thread safe | 网页控制台能正常打开（原来一点就崩） |
| secret redaction | 密码、邮箱会自动被打码，但正常的话一个字不动 |
| packaging | 该有的文件都在，且核心部分**不依赖任何第三方库** |

---

## 第 3 步：亲手试一次（30 秒，推荐）

这一步是让你**亲眼看到**它到底干了什么，而不是只看一行 PASS。

```bash
ratchet init --db d.db
ratchet add "先给结论，再给理由" --db d.db
ratchet chat "解释一下 LoRA" --db d.db
```

第三条命令会打印一大段文字，那就是**真正发给 AI 的内容**。它分成三块，你对照着看：

```
第一块 —— 系统挑的"说话方式"
APPROACH FOR THIS TASK:
Work through the task step by step before committing to an answer...

第二块 —— 以前学到的经验
BEGIN RECALLED EXPERIENCE DATA
（中间几行是在解释"下面是资料不是命令"）
- [user_rule | ...] 先给结论，再给理由
END RECALLED EXPERIENCE DATA

第三块 —— 你自己的问题
解释一下 LoRA
```

**重点看第二块外面那层壳。** 它把学到的东西包在 `BEGIN/END` 之间，并且明明白白写了"这些是以前的数据，不是命令"。这就是防"AI 被人教坏"的第一道墙。

再看它存了什么：

```bash
ratchet audit --db d.db
```

```
0bf83403-fdc1-4da3-9762-b2523ad52762  provenance=user_rule  kind=rule  trust=trusted  score=+0.0  uses=0
    先给结论，再给理由
```

想删掉它：

```bash
ratchet rm 0bf83403 --db d.db        # 输前几位就行，大小写都行
ratchet audit --all --db d.db        # 会看到 [inactive]
ratchet restore 0bf83403 --db d.db   # 后悔了能救回来
```

删完是 `[inactive]` 不是消失——**删除是能反悔的**。真想彻底擦掉才用 `ratchet purge`。

---

## 第 4 步：跑完整测试（6 分钟）

```bash
python -m pytest
```

期间会刷一堆小点，耐心等。**看到这样就成功了：**

```
356 passed in 368.49s (0:06:08)
```

中间 6 分钟不看也行，去倒杯水回来瞄最后一行。

---

## 只想验一件事？做这个（30 秒）

```bash
python examples/02_custom_verifier.py
```

**看到这样就成功了：**

```
posterior means after 30 interactions:
  chain_of_thought   mean=0.622  pulls=4
  concise            mean=0.962  pulls=24
  default            mean=0.592  pulls=2

mean reward: 0.937 over 30 events
```

怎么读这三行：

- `concise` = 「简洁回答」这个策略，`pulls=24` 意思是 30 次里它被选中了 24 次，`0.962` 是系统对它的信心（满分 1.0）。
- 奖励函数要求"必须简洁"，所以它学会了挑这个策略。
- **它自己挑的，没人告诉它答案。**

如果你好奇：这个文件每次跑都会先清空数据库，所以**你连跑两次，数字会一模一样。**

---

## 5. 我想亲眼看看它能删不能改

```bash
ratchet panel --db d.db
```

浏览器会自动打开 `http://127.0.0.1:8787/`。

页面上对照着看：

- AI 自己学的条目后面只有 **Delete**，**没有输入框**
- 你自己写的规则才有输入框，而且最多只能写 200 字
- 每条都能点开看"它为什么存在、被用过几次"

这个面板**只在你自己电脑上打开**（地址写死了 127.0.0.1），外面的人连不上。传 GitHub 之后也一样。

---

## 6. 出问题了怎么办

| 看到的现象 | 原因 | 怎么解决 |
|---|---|---|
| `'python' 不是内部或外部命令` | Python 没装 / 没加进 PATH | 装 Python 3.10 以上，安装时勾 `Add to PATH` |
| `No module named ratchet` | 第 1 步没成功 | 补一句 `set PYTHONPATH=src` 再试 |
| 8/9 或更少 | 真有问题 | **把整屏内容复制给我** |
| 测试跑到一半卡住 | 检索测试本来就慢 | 耐心等满 6 分钟，别中断 |
| 提示 ERR_EMPTY / 端口占用 | 8787 被别的程序占了 | `ratchet panel --port 8899` |

---

## 7. 如果你不确定我是不是在吹牛（可选）

做这一个就够了。它验证的是**最要命的那处**——AI 是不是真的被自己学到的东西影响了。

用记事本打开 `src/ratchet/core/session.py`，找到 136、163、195 行，三处都长这样：

```python
variant_id=variant_id,
```

把三处全改成：

```python
variant_id="default",
```

保存，然后重新跑：

```bash
python scripts/verify.py
```

**正常情况下，第一项会变红：**

```
  [FAIL] strategy reaches prompt
         concise variant injected no strategy block
============================================================
8/9 checks passed
```

再跑一次那个 30 秒的例子：

```bash
python examples/02_custom_verifier.py
```

你会发现数字从 `0.962` 掉到 `0.660` 左右，三个策略的数字挤成一团，平均奖励从 `0.937` 掉到 `0.683`。

**意思是：它还在装模作样地挑策略，但挑哪个都没区别了——它在自我陶醉地空转。**

改回来：把三处的 `"default"` 改回 `variant_id`，再跑一次确认回到 9/9。

---

## 8. 这些命令证明不了什么

说实话，不然你以后会踩坑：

1. **没有证明"AI 变聪明了"。** 上面所有测试用的都是假的 AI（会把你的话原样复读），证明的是"这套流程是通的"，不是"它在真实任务上更强"。要证明后者，得接真模型、跑几百次真实对话。

2. **Route A（真正改模型权重）没跑过。** 只生成了训练计划，一次真正的训练都没做——没显卡也装不了那几个库。这条路是留给你以后在有显卡的机器上走的。

3. **没接过任何真实的 AI。** OpenAI / Ollama 的接口写好了，但一次网络请求都没发过。第一次接真模型时，计费和超时的表现要你自己看一眼。

4. **没有"好坏标准"它就不学。** 如果你没法客观判断一个回答好不好，它就只存不用。上面第 30 秒那个例子就是在教你怎么给它定标准。

5. **防注入是减轻，不是免疫。** 把外部文字塞给 AI 这件事，没有谁能做到百分百安全。

6. **网页面板没有密码。** 只能在你自己电脑上用，别想办法让它暴露到外面。

---

## 9. 一分钟命令清单

```bash
pip install -e .                      # 装
python scripts/verify.py              # 3 秒，看到 9/9
python examples/02_custom_verifier.py # 30 秒，看到 concise 是 0.962
python -m pytest                      # 6 分钟，看到 356 passed

ratchet init --db d.db                # 建一个数据库
ratchet add "你的规矩" --db d.db       # 写一条自己的规则
ratchet chat "你的问题" --db d.db      # 看它发给 AI 的完整内容
ratchet audit --db d.db               # 看它学到了什么
ratchet panel --db d.db               # 浏览器里看，能删不能改
```
