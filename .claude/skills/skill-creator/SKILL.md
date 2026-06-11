---
name: skill-creator
description: Create and improve skills for Claude Code. Use whenever the user wants to build a new skill from scratch, improve an existing one, or capture a workflow as a reusable skill. Trigger on phrases like "make a skill", "turn this into a skill", "create a skill for X", "update this skill", or when the user completes a workflow and wants to save it.
---

# Skill Creator

A skill for creating and improving Claude Code skills.

## The Core Loop

1. Understand what the skill should do
2. Write or update the SKILL.md
3. Test it on realistic prompts
4. Improve based on results
5. Repeat until satisfied

Jump in wherever the user is in this process.

---

## Step 1: Capture Intent

If the user just finished a workflow, extract from the conversation:
- What tools were used and in what order
- What inputs and outputs were involved
- Any corrections the user made along the way

Otherwise, ask:
1. What should this skill enable Claude to do?
2. When should it trigger? (what phrases or contexts)
3. What's the expected output?

---

## Step 2: Write the SKILL.md

Every skill is a folder with a `SKILL.md` at the root. Optional subfolders:
- `scripts/` — reusable helper scripts
- `references/` — docs loaded as needed
- `assets/` — templates, fonts, files used in output

### Frontmatter (required)

```
---
name: skill-name
description: When to trigger and what it does. Be specific and slightly pushy — Claude tends to undertrigger, so include multiple phrasings of when to use this skill.
---
```

### Body

Write in plain markdown. Use imperative instructions. Explain the *why* behind each step — Claude follows reasoning better than rigid rules.

**Good pattern — define output format explicitly:**
```markdown
## Output format
Always use this structure:
# Title
## Summary
## Details
## Next steps
```

**Good pattern — include a worked example:**
```markdown
## Example
Input: user asks to refactor auth module
Output: creates a plan, applies changes file by file, runs tests
```

### Keep it lean
- Under 500 lines for SKILL.md
- If it grows large, split into `references/` files and link to them
- Remove anything that isn't pulling its weight

---

## Step 3: Test It

Come up with 2–3 realistic prompts a real user would type — not abstract, but specific with file paths, context, and natural phrasing. Share them with the user first.

Run each prompt yourself following the skill's instructions. For each test, note:
- Did it follow the skill correctly?
- Was the output what the user wanted?
- Any steps that were wasteful or wrong?

---

## Step 4: Improve

After reviewing results, revise the skill. Good improvement principles:

- **Generalize from feedback** — don't hardcode fixes for the exact test case; write instructions that work broadly
- **Explain the why** — instead of `ALWAYS do X`, explain why X matters so Claude can apply good judgment
- **Bundle repeated work** — if every test run independently wrote the same helper script, add it to `scripts/` and reference it from the skill
- **Cut what doesn't help** — if a section didn't improve outputs, remove it

---

## Step 5: Package

When the skill is done, package it into a `.skill` file for distribution:

```bash
python -m scripts.package_skill path/to/skill-folder output-dir/
```

This creates a zip-based `.skill` file the user can install.

---

## Writing Tips

- Start with a draft, then read it with fresh eyes and tighten it
- Use theory of mind — imagine Claude reading this cold and ask if the instructions make sense
- Prefer prose instructions over rigid checklists for complex judgment calls
- Use checklists only for deterministic steps where order matters
- If you find yourself writing `MUST` or `NEVER` in caps, ask if you can explain the reasoning instead