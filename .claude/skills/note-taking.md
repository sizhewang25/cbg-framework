# Note-Taking Skill

Save research insights, progress, and findings to the `notes/` folder with daily timestamps.

## Instructions

When the user asks to take notes, save insights, or record progress:

1. **File naming**: Use the format `notes/YYYY-MM-DD-<topic>.md` (e.g., `notes/2026-01-28-cbg-analysis.md`)
2. **If a note for today's topic exists**: Append to it with a new timestamp section
3. **If no note exists**: Create a new file

## Note Format

```markdown
# <Topic Title>

## <YYYY-MM-DD HH:MM> - <Brief Section Title>

<Content here>

### Key Points
- Point 1
- Point 2

### Next Steps
- [ ] Task 1
- [ ] Task 2

---
```

## Guidelines

- Keep notes concise but informative
- Use headers to organize sections
- Include code snippets with proper markdown formatting when relevant
- Link to relevant files in the codebase using relative paths
- Tag insights with categories when useful: `[DATA]`, `[ALGORITHM]`, `[GAP]`, `[IDEA]`, `[TODO]`
- Always confirm with the user what was saved

## Example Usage

User: "Take a note about the CBG calibration gap we discovered"

Action: Create/append to `notes/2026-01-28-cbg-analysis.md` with timestamped entry
