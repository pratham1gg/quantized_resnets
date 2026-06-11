---
name: notebook-writer
description: Write, fix, and edit Jupyter notebooks that read like a human wrote them. Trigger on any request involving .ipynb files — "write a notebook", "fix this notebook", "add cells to the notebook", "make the notebook look natural", "clean up the notebook", or when the user points to an .ipynb file and asks for changes.
---

# Notebook Writer

Write and fix Jupyter notebooks so they read like a real person wrote them — not an AI. The goal is notebooks that look like a researcher or engineer sat down, explored something, and wrote up their thinking as they went.

## Core Principles

### 1. Write like a person, not a textbook

Human notebooks have a voice. They use first person, contractions, and casual transitions. They don't over-explain obvious things or narrate every import.

**Bad (AI-sounding):**
```markdown
## Data Loading
In this section, we will load the dataset and perform initial preprocessing steps to prepare it for model training.
```

**Good (human-sounding):**
```markdown
## Loading the data
Let's load ImageNet and set up the transforms. We'll use standard normalization values.
```

Key patterns to follow:
- Use "let's", "we", "I" naturally — not in every sentence
- Skip the preamble. Don't announce what you're about to do in a markdown cell right before doing it in code
- Vary sentence length. Mix short punchy observations with occasional longer explanations
- Use markdown headers sparingly — not every 2 cells. Humans don't section everything
- It's okay to have 3-5 code cells in a row without any markdown in between
- When you do write markdown, keep it to 1-3 sentences most of the time. Save longer text for actual insights or methodology explanations

### 2. Code should look hand-written

Human code in notebooks is practical, not showcase code. It's clean but not obsessively so.

Patterns:
- **Short variable names where context is obvious**: `model`, `loader`, `acc`, `loss`, `lr` — not `training_accuracy_score`
- **Inline comments only when non-obvious**: a comment on a tricky tensor reshape is good; a comment saying `# load the model` above `model = load_model()` is not
- **Don't wrap every block in try/except** — humans don't do that in notebooks
- **Print intermediate results** — humans check shapes, peek at values, print `len(dataset)`. Sprinkle these in naturally
- **Imports at the top** in one or two cells, grouped loosely (stdlib, then third-party, then local) — but don't be obsessive about it. If you need a one-off import later, just put it where you use it
- **Magic commands are fine**: `%matplotlib inline`, `%%time`, `%load_ext` — humans use these
- **Variable reuse across cells** — humans build on previous cells, they don't re-derive everything

### 3. Structure should feel organic

Real notebooks have a natural flow — they're not perfectly symmetric or evenly sectioned.

- Start with imports and setup (1-2 cells, no markdown header needed for this)
- The meat of the notebook should flow: do something, look at results, adjust, continue
- Not every section needs the same depth. If something is straightforward, one cell is fine. If it's interesting, spend more cells on it
- End with results/conclusions but keep it brief — a few sentences, maybe a summary table or plot
- **Don't number your sections** (1.1, 1.2, etc.) — humans rarely do this in notebooks
- Mix cell sizes: some cells are 2 lines, some are 15. Don't make every cell the same length

### 4. Markdown formatting

- Use `##` for main sections, `###` for subsections. Never use `#` (title-level) inside the notebook except maybe the very first cell
- **Bold** for emphasis on key numbers or findings, not for structure
- Use bullet points occasionally, not as the default way to present everything
- LaTeX math is fine when it clarifies: `$\alpha = 0.01$` or a loss formula. Don't over-use it
- Tables are great for comparing results — use them instead of printing DataFrames when presenting final numbers
- No horizontal rules (`---`) between sections — let headers do the work

### 5. Plots and visualization

- Keep plot code straightforward — `plt.plot()`, `plt.bar()`, basic seaborn
- Set figsize to reasonable defaults (8,5 or 10,6 — not tiny, not huge)
- Always label axes with units where applicable (e.g., "latency in milliseconds")
- Use `plt.tight_layout()` or `fig.tight_layout()` before `plt.show()`
- Title plots descriptively but briefly: "Top-1 accuracy vs quantization bit-width"
- Don't over-style. Light grid (`plt.grid(alpha=0.3)`) is fine. Skip the seaborn themes unless the user asks

## When Fixing Existing Notebooks

1. **Read the entire notebook first** using the Read tool
2. **Identify problems**: broken cells, missing imports, wrong variable names, outdated APIs, bad outputs
3. **Fix in-place** using NotebookEdit — don't rewrite the whole notebook unless asked
4. **Preserve the author's voice** — if they write casually, keep it casual. Match their style
5. **Don't add markdown cells** to an existing notebook unless there's a clear gap. Fix what's there
6. **Clear stale outputs** if the code has changed and outputs are misleading

## When Writing New Notebooks

1. **Ask yourself**: what's the one thing this notebook is about? Keep that focus
2. **Start with a title cell** — markdown with `##` title and 1-2 sentences of context. Not an abstract, just enough so someone opening the notebook knows what it does
3. **Imports next** — one code cell, no markdown header above it
4. **Build up naturally** — load data, do the thing, show results. Like you're working through the problem
5. **Include sanity checks** — shape prints, small sample visualizations, assertions. Humans do this
6. **End clean** — summarize findings or show final results. 2-5 sentences max

## Anti-patterns to Avoid

These are telltale signs of AI-generated notebooks. Never do these:

- "In this notebook, we will explore..." (announcing intentions)
- "As we can see from the output above..." (narrating what's visible)
- "Let's proceed to the next step" (filler transitions)
- Every cell having a markdown cell above it explaining what it does
- Perfectly uniform cell sizes
- Over-commented code (`# Import numpy` above `import numpy as np`)
- Wrapping everything in functions when sequential code is clearer
- Adding error handling in a notebook context (just let it fail — that's what notebooks are for)
- Numbered sections (1., 2., 3.) or TOC-style headers
- Concluding with "In this notebook, we have successfully..."
