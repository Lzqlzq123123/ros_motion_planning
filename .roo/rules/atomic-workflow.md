# 原子任务执行协议 (ASCM)

你必须严格遵守以下工作流，不得跳过任何步骤：

1. **任务读取**：当用户提出新需求时，必须首先读取根目录下的 task_planning_prompt.md 模板，完成结构化的任务拆分。
- 1.1 分析输入：读取用户需求、当前 CURRENT_CODEBASE_STATUS 和 FILE_STRUCTURE。
- 1.2 初始分析：首次读取代码前，将当前系统架构和逻辑状态的分析摘要写入 .agent/global_log.md。
- 1.3 原子拆分：使用 task_planning_prompt.md 生成 backlog.json。
- 1.4 用户准入：生成的 backlog.json 必须由用户审核确认。如果项目根目录存在 `backlog.json`，你必须优先按顺序处理其中的原子任务。
2. **环境支架**：
   - 每次修改代码前，读取 `.agents/global_log.md` 了解前置任务的 context。
   - 每次修改代码后，必须严格按照以下格式追加到 `agent/global_log.md`：
   """
   ## [TASK_ID]
   - **Changes**: [文件名]: [起始行号]-[结束行号] -> [改动简述]
   - **Line Stats**: +[新增行数], -[删除行数]
   - **Errors**: [报错及修复]
   - **Context**: [后续任务需知的变量、接口等上下文]
   """
   - 每次用户反馈的错误都要记录

3. **Git & 物理自愈**：
   - **成功**：若原子任务测试通过且无报错，自动执行 `git add` 和 `git commit`（格式：feat: [ID] description）。
   - **失败**：若连续修复两次仍失败，立即执行 `git reset --hard HEAD` 并停下来请求人工介入。
4. **持久化日志**：每次任务结束，将本次修改的 ID、逻辑变更、新增变量名写入 `.agents/global_log.md`。

