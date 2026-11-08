# 怎么把这个项目传到 GitHub 上

> 按"你从没用过 GitHub"来写的。每一步都告诉你点哪里、输什么。
> **我已经把最难的部分做完了**——本地仓库建好了，文件全部整理好了，第一版也已提交。
> 你只需要做 **4 件事**，大概 10 分钟。

---

## 先搞清楚三件事

**1. GitHub 是什么？**
一个专门放代码的网盘。放上去（公开的话）全世界都能看到，也能给链接给别人。对企业面试、作品集都有用。

**2. 上传的其实不是"文件"，而是一条时间线。**
每次你保存一次改动，就叫一个 commit（提交）。Git 记录的是"第 1 版 → 第 2 版 → 第 3 版"这条线。好处是你随时能回到任何一版，**这也是它比网盘强的地方**。我已经帮你做好了第 1 版。

**3. 我已经帮你挡掉了私密文件。**
项目里的 `.gitignore` 会自动跳过这些：

| 被挡掉的东西 | 为什么必须挡 |
|---|---|
| `*.db` 数据库 | **里面是你真实的对话内容、代码、偏好**。传上去等于公开日记 |
| `.env` 配置文件 | 放着 API key 和口令 |
| `.venv/` 虚拟环境 | 几百 MB 的库，没必要传 |
| `__pycache__`、各种缓存 | 垃圾文件 |

所以你放心传，隐私这一层我已经替你把过一道了。上传前还有一步自检，在第 5 节。

---

## 第 1 步：注册 / 登录

打开 <https://github.com>，右上角 Sign in（有账号就登录，没有就 Sign up，用邮箱就行）。

---

## 第 2 步：新建一个空仓库

1. 登录后，点页面**右上角那个 `+` 号** → **New repository**
2. 按下图填：

| 项目 | 填什么 | 注意 |
|---|---|---|
| Repository name | `ratchet` | 只能用英文、数字、横线 |
| Description（选填） | `Local-first self-improvement engine for any LLM` | 一句话，别人搜得到 |
| Public / Private | 选 **Public** | 想先自己看就选 Private，以后能改公开 |
| **Add a README file** | **一定不要勾** ⚠️ | 我们已经有 README 了，勾了会冲突 |
| **Add .gitignore** | **不要勾** ⚠️ | 我们自己的更完整 |
| **Choose a license** | **不要勾** ⚠️ | LICENSE 文件已经有了 |

> **只要这三个框是空的，就对了。** 页面上应该写 "You have selected an empty repository" 之类的话。

3. 点绿色的 **Create repository**

---

## 第 3 步：把两行命令粘到终端里

> **先看这里（30 秒）**：第一次提交时，作者名字我先填了 `Maverick <maverick@users.noreply.github.com>`。
> 如果你的 GitHub 用户名不叫这个，先改成你自己的，这样提交记录才挂在你的名下：
>
> ```bash
> git config user.name "你的GitHub用户名"
> git config user.email "你的邮箱"
> git commit --amend --reset-author --no-edit
> ```
>
> 名字就是 Maverick 的话，跳过这段。

创建完会跳到一个页面，上面有一堆命令。**别管它写的那些，你只需要下面两行。**

回到你那个黑窗口（在 `ratchet` 文件夹里），粘进去：

```bash
git remote add origin https://github.com/你的用户名/ratchet.git
git push -u origin main
```

> 第 1 行里的 `你的用户名` 要换成你自己的 GitHub 用户名。
> 比如你叫 `zhangsan`，那就是 `https://github.com/zhangsan/ratchet.git`。

回车之后会问你账号密码：

- **Username**：你的 GitHub 用户名
- **Password**：**这里不能填登录密码！** 见下面第 4 步

---

## 第 4 步：生成"专用密码"（Token）

GitHub 从 2021 年起不让用登录密码推代码了，要用一个叫 **Personal Access Token** 的东西。做一次就行。

1. 点右上角**你的头像** → **Settings**
2. 左边最下面 **Developer settings** → **Personal access tokens** → **Tokens (classic)**
3. 右上角 **Generate new token** → **Generate new token (classic)**
4. 按下图填：

| 项目 | 填什么 |
|---|---|
| Note（备注） | 随便写，比如 `ratchet` |
| Expiration（有效期） | 选 **90 days**（到期重新生成就行，更安全） |
| 勾选权限 | 只勾 **`repo`**（第一个大勾，会把下面几个子项一起勾上） |

5. 拉到最下面点 **Generate token**
6. **页面上会显示一串 `ghp_xxxx...` 的字符串，立刻复制保存下来** ⚠️

   > 这串东西**只显示这一次**，关掉页面就再也找不回来了（只能重新生成一个）。

7. 回到黑窗口重新执行：

```bash
git push -u origin main
```

这次 **Username 填你的用户名，Password 粘贴那串 `ghp_...`**。

> 粘贴的时候屏幕上**不会显示任何字符**，不是卡了，是故意的。粘完直接回车。

---

## 搞定。看到这样就成功了

```
Enumerating objects: ...
Writing objects: 100% ...
To https://github.com/你的用户名/ratchet.git
 * [new branch]      main -> main
```

现在刷新 GitHub 页面，**F5**。页面会自动变成项目首页，README 的内容显示在下面。🎉

---

## 第 5 步：传完之后，请做完这 5 项自检

对着你的 GitHub 仓库页面检查（**这一步别省**）：

- [ ] 页面下半部分能看到 README 的内容（不是空白）
- [ ] 文件列表里**没有一个 `.db` 结尾的文件** ← 这是你的隐私，最重要的一条
- [ ] 文件列表里**没有 `.env`**（可以有 `.env.example`，那个是模板，没密码）
- [ ] 有 `LICENSE` 文件
- [ ] 搜一下确认没有 `.venv`、`__pycache__`、缓存目录混进去

任何一条不对，**立刻把仓库设成 Private，然后告诉我**。

---

## 如果不会用命令行：网页拖拽法

不想碰黑窗口也能传，但有缺点。

1. 第 1、2 步一样操作
2. 在空仓库页面上，点 **uploading an existing file** 这个链接
3. 把 `ratchet` 文件夹**里面的东西**拖进去（注意是里面的内容，不是文件夹本身）
4. 最下面点 **Commit changes**

**缺点（你要知道）：**

| 问题 | 说明 |
|---|---|
| 拖不进去隐藏文件 | `.gitignore`、`.github/` 是隐藏的，Windows 默认看不到也不会拖。缺了 `.gitignore`，**你的数据库就有被传上去的风险** |
| 拖不进去空文件夹 | 无所谓，本项目没有需要保留的空文件夹 |
| 没有版本历史 | 只有一个大 commit，以后不好回退 |
| 超过 100 个文件会卡 | 本项目 74 个，勉强够用 |

**所以还是推荐命令行法。** 如果你实在要用拖拽，拖完之后请务必做第 5 节那 5 项自检。

---

## 常见问题

**Q：push 的时候提示 `failed to push some refs` / `non-fast-forward`？**
说明远程仓库里有东西本地没有（通常是第 2 步不小心勾了 README）。先执行：

```bash
git pull --rebase origin main
git push -u origin main
```

> 不要用 `--force`。会把别人的东西冲掉，养成习惯以后迟早出事。

**Q：提示 `Authentication failed`？**
密码填的是登录密码。要用 Token（第 4 步那串 `ghp_`）。

**Q：提示文件超过 100MB？**
单个文件不能超过 100MB。本项目不会有这种情况，如果发生了说明你把某个大文件（比如缓存目录）加进来了，告诉我。

**Q：我想先自己偷偷看，以后再公开？**
第 2 步选 Private。之后随时能在 Settings → Danger Zone → Change visibility 改成公开。

**Q：以后我改了代码，怎么更新？**
在 `ratchet` 文件夹里执行这三条：

```bash
git add -A
git commit -m "改了什么，一句话"
git push
```

**Q：Token 过期了怎么办？**
重新走一遍第 4 步，生成新的，覆盖掉旧的。

---

## 传完之后可以做的事（选填）

**写一句简介**：仓库页面右上角那个小齿轮（⚙️ About），把 Description 填上，再加几个 Topics（话题标签）：

```
llm  ·  reinforcement-learning  ·  rag  ·  prompt-engineering  ·  local-first  ·  python
```

别人搜这些词的时候能搜到你。

**验证别人能不能装上**（换台电脑或者用手机热点试）：

```bash
pip install git+https://github.com/你的用户名/ratchet.git
```

装得上就说明完美。

**README 里的链接要不要改？**
`pyproject.toml` 里写的项目地址是占位的 `github.com/ratchet-ai/ratchet`。想改成你自己的话，搜这个文件里的 `ratchet-ai` 换成你的用户名就行（不影响能不能用）。

---

## 最后：验收和上传是两件事

| 你想做什么 | 看哪个文件 |
|---|---|
| 确认这套东西真的能用 | 👉 [`怎么验收这个项目`](./how_to_verify.md) |
| 传到 GitHub 上 | 👉 就是这份 |
| 搞懂它的原理 | [`架构说明`](./architecture.md) |
| 搞懂"机器能增、人只能删" | [`治理设计`](./governance.md) |
