
# Claude Code Best Practices (Practical Guide)

## 🧠 Core Mindset
Treat Claude Code as a **teammate**, not a one-shot tool.

- Provide context
- Guide behavior
- Iterate and improve over time

---

## 📌 1. Prompt Structure (Default Template)

Always include:

- **Goal**: what to build / fix
- **Context**: relevant files, errors, examples
- **Constraints**: rules, architecture, safety limits
- **Done Criteria**: what success looks like

**Example:**

> Goal: Fix training bug  
> Context: train.py, error log  
> Constraints: no refactor, keep API  
> Done: loss decreases and test passes

---

## 🧠 2. Thinking Level (Claude's Extended Reasoning)

Claude Code supports extended thinking for complex tasks:

- **Simple** → direct answers, formatting, obvious fixes
- **Medium** → debugging, single-file changes
- **High** → multi-file features, refactoring planning
- **Very High** → architecture decisions, complex debugging

Use `💭` thinking tags for complex reasoning when needed.

---

## 🪜 3. Plan Before Coding

For non-trivial tasks:

- Ask Claude to **propose a plan first**
- Or explicitly say "plan before coding"

**Workflow:**
1. Understand
2. Plan
3. Confirm (wait for user approval)
4. Implement

---

## 📄 4. Use CLAUDE.md (Persistent Rules)

Store reusable rules in `CLAUDE.md` at repo root:

- project structure
- build/test commands
- coding constraints
- definition of "done"

**Best practices:**
- Keep it short and actionable
- Add rules only after repeated mistakes
- Use `.claude/CLAUDE.md` for team-shared rules
- More specific files (subdirectories) override global ones

---

## ⚙️ 5. Configuration (Claude Code Settings)

Claude Code uses:
- `~/.claude/settings.json` → personal defaults
- `.claude/settings.json` → project settings
- `.claude/settings.local.json` → local overrides (gitignored)

**Control:**
- model selection (sonnet / opus / haiku)
- thinking budget
- permissions (allow/ask/deny for tools)
- MCP server configs

---

## 🔒 6. Safety & Permissions

Start strict:

- Use permission modes: `--dangerously-skip-permissions` only for trusted repos
- Prefer `--permission-mode=acceptEdits` or `bypassPermissions`

**Permission levels:**
- `deny` → block specific tools/patterns
- `ask` → prompt for approval
- `allow` → auto-execute

**Recommended initial config:**
```json
{
  "permissions": {
    "allow": ["Read", "Bash(git:*)", "Bash(npm:*)"],
    "ask": ["Write", "Edit", "Bash(rm:*)"],
    "deny": ["Bash(sudo:*)"]
  }
}
```

---

## 🔍 7. Verification Loop (Critical)

Always require Claude Code to:

- write or update relevant tests
- run existing tests
- check for syntax errors
- review the diff before finishing

Claude should:
- verify behavior matches intent
- check for unintended regressions
- confirm done criteria are met

---

## 🔎 8. Code Review

Use Claude's ability to review changes:

- Ask "review this change" with the diff
- Check for:
  - bugs or logic errors
  - unintended side effects
  - risky patterns (performance, security)

---

## 🔌 9. MCP (External Context)

Use MCP servers when:

- data is outside the repository
- data changes frequently
- tools are needed (API calls, database queries, log analysis)

**Available MCP servers** (via `claude mcp add`):
- Filesystem
- GitHub
- Slack
- Custom servers

**Best practice:**
- Start with built-in tools
- Add MCP servers only when needed
- Document MCP usage in CLAUDE.md

---

## 🧩 10. Skills (Reusable Workflows)

Use or create skills when tasks repeat:

- debugging patterns
- PR review checklist
- log analysis
- test generation

**Rule:** If you repeat a prompt 3+ times → consider making a skill

**Skill structure:**
- `SKILL.md` with clear description
- Keep input/output well-defined
- Simple scope, composable

Use `~/.claude/skills/` for personal skills, or repo `.claude/skills/` for team skills.

---

## 🔁 11. Automation (Bash Commands)

Use bash commands for stable, repeatable workflows:

Good candidates:

- running test suites
- linting and formatting
- build processes
- generating summaries

**Best practice:** Wrap in `package.json` scripts or Makefile, then Claude can just run `npm run test`

---

## 🧵 12. Session Management

- **One task = one conversation thread**
- Keep related work in same session
- Use `/clear` to reset when switching tasks
- Use project-specific sessions: `claude --print --session "project-name"`

**Avoid:**
- mixing unrelated tasks in one session
- extremely long sessions without breaks (context can drift)

---

## ⚠️ Common Mistakes

| ❌ Mistake | ✅ Fix |
|------------|--------|
| No planning for complex tasks | Ask for plan first |
| Too much in prompt instead of CLAUDE.md | Move persistent rules to CLAUDE.md |
| Giving full permissions too early | Start with `ask` for write/edit |
| No test / verification | Add verification step to workflow |
| Treating Claude as one-shot tool | Iterate and refine |
| One session per project (instead of per task) | Create new sessions for unrelated tasks |
| Ignoring Claude's questions | Answer them—they're asked for a reason |

---

## 🎯 Final Rule

Claude Code works best when:

- context is clear (CLAUDE.md is populated)
- constraints are explicit (permissions configured)
- workflow is structured (plan → confirm → implement → verify)
- verification is enforced (tests, linting, review)

---

## 📚 Quick Reference Card

```yaml
Goal:      # What needs to be done
Context:   # Files, errors, relevant info
Constraints: # What NOT to touch
Done:      # Success criteria

Workflow: Plan → Confirm → Implement → Verify

Config:   .claude/settings.json
Rules:    CLAUDE.md (repo root)
Skills:   .claude/skills/
MCP:      claude mcp add <name> <command>
Session:  claude --session <name>

