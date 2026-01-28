# Task Focus Skill

Quickly load context for a specific task from the `tasks/` folder to resume work efficiently, even after session reset.

## Instructions

When the user says **"focus on [task-name]"** or **"load task [task-name]"**:

1. **Find the task folder**: Look in `tasks/<task-name>/`
2. **Read all context files** in this order:
   - `README.md` - Task overview, background, and requirements
   - `todos.md` - Current todo list and progress tracking
   - `progress.md` - Detailed progress log with timestamps
   - `context.md` - Technical context, code references, key findings
3. **Summarize** the current state to the user
4. **Identify** the next actionable item from todos

## Task Folder Structure

```
tasks/
└── <task-name>/
    ├── README.md      # Background, goals, requirements, success criteria
    ├── todos.md       # Checklist of tasks (use [x] for done, [ ] for pending)
    ├── progress.md    # Timestamped progress log (append new entries)
    └── context.md     # Technical context, code refs, key findings, decisions
```

## File Templates

### README.md
```markdown
# Task: <Task Title>

## Background
<Why this task exists, problem statement>

## Goals
- Goal 1
- Goal 2

## Requirements
- Requirement 1
- Requirement 2

## Success Criteria
- [ ] Criterion 1
- [ ] Criterion 2

## Related Files
- [file1.py](../../path/to/file1.py)
- [file2.ipynb](../../path/to/file2.ipynb)
```

### todos.md
```markdown
# Todos

## In Progress
- [ ] Current task

## Pending
- [ ] Next task 1
- [ ] Next task 2

## Completed
- [x] Finished task 1
```

### progress.md
```markdown
# Progress Log

## YYYY-MM-DD HH:MM - <Title>
<What was done, findings, decisions made>

---
```

### context.md
```markdown
# Technical Context

## Key Findings
- Finding 1
- Finding 2

## Code References
- `path/to/file.py:123` - Description

## Decisions Made
- Decision 1: Rationale

## Open Questions
- Question 1?
```

## Commands

| User Says | Action |
|-----------|--------|
| "focus on X" | Load task X context |
| "update task todos" | Update todos.md |
| "log progress" | Append to progress.md |
| "add context" | Append to context.md |
| "create task X" | Create new task folder with templates |
| "list tasks" | Show all tasks in tasks/ folder |

## Example

User: "focus on cbg-feasibility"

Claude:
1. Reads `tasks/cbg-feasibility/README.md`
2. Reads `tasks/cbg-feasibility/todos.md`
3. Reads `tasks/cbg-feasibility/progress.md`
4. Reads `tasks/cbg-feasibility/context.md`
5. Summarizes: "Task: CBG Feasibility. Current status: X. Next action: Y"
