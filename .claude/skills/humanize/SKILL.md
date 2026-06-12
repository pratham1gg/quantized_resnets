---
name: humanize
description: Remove AI-sounding comments, over-explained docstrings, and robotic code patterns to make code look human-written. Trigger on "humanize", "make it look human", "remove AI comments", "clean up AI code", "de-AI", "make code natural", "remove verbose comments", or when the user points at code and says it looks too AI-generated.
---

# Humanize

Strip AI fingerprints from code — the verbose comments, the over-explained docstrings, the robotically-named variables, the narrated control flow. Make it read like a human wrote it on the first pass.

## What to target

### Comments to kill outright

These add zero information. Delete them:

- **Narration comments** that restate what the line does: `# Initialize the list`, `# Loop through items`, `# Return the result`, `# Check if value is None`
- **Section headers on obvious blocks**: `# --- Data Loading ---`, `# === Model Setup ===`
- **Filler comments**: `# As needed`, `# For clarity`, `# This is important`
- **"This function" openers**: `# This function calculates...`, `# Helper function to...`, `# This method is responsible for...`
- **TODO-style noise**: `# TODO: Add error handling here` (if there's no actual TODO worth keeping)
- **Redundant type annotations in comments** when the code has type hints

### Comments to shorten

Some comments have real info buried under AI verbosity. Extract the core:

- `# Calculate the weighted average of all scores using the provided weights` → `# weighted avg` or just delete if the function name says it
- `# Ensure that the input tensor has the correct dimensions before proceeding` → `# check dims` or delete if an assert follows
- Multi-line docstrings on simple functions → one line or nothing

### Docstrings

- If a function name + signature already tells you what it does, the docstring is noise. Remove it.
- If a docstring restates the parameter names and types that are already in the signature, remove it.
- Keep docstrings only when they explain **why** or document non-obvious behavior.
- A good docstring is 1 line. A bad docstring is a paragraph that restates the function name.

### AI vocabulary

These words/phrases are AI tells. Replace or remove:
- "leverage" → "use"
- "utilize" → "use"  
- "facilitate" → "help" or remove
- "ensure that" → just do the thing
- "in order to" → "to"
- "is responsible for" → does/handles
- "comprehensive" → remove or "full"
- "robust" → remove
- "seamlessly" → remove
- "Note:" / "Important:" prefixes on obvious things → remove

### Code patterns to simplify

- **Over-named variables**: `image_classification_result` → `result` or `pred` if context is clear
- **Unnecessary intermediate variables** used once on the next line — inline them
- **Over-structured try/except** with generic comments — simplify if the error handling is trivial
- **Gratuitous blank lines** between every logical "step" — humans don't triple-space everything
- **Redundant `else` after `return`** — just dedent

### Notebook cells (.ipynb)

- Remove markdown cells that are just AI narration: "Now let's...", "In this section, we will...", "Let's go ahead and..."
- Keep markdown cells that have actual section titles or explanations of methodology
- Remove `print("Step X complete")` type progress narration unless it serves a real purpose
- Remove cells that just print shapes/types for "verification" unless the user clearly wants them

## How to apply

1. **Read the target** — the user gives a file or directory. If a directory, find all `.py`, `.ipynb`, `.js`, `.ts`, `.jsx`, `.tsx`, `.java`, `.cpp`, `.c`, `.go`, `.rs` files.

2. **Work file by file** — for each file:
   - Read the full file
   - Identify all AI patterns (comments, docstrings, naming, structure)
   - Apply edits using the Edit tool, grouping nearby changes when possible
   - For notebooks, use NotebookEdit

3. **Preserve real comments** — not every comment is AI slop. Keep comments that:
   - Explain a workaround or hack (`# PIL has a bug with 16-bit PNGs, convert first`)
   - Document a non-obvious invariant (`# must be called before fork()`)
   - Contain URLs or references
   - Are clearly human-written (informal, terse, sometimes sweary)
   - Explain **why**, not **what**

4. **Don't refactor logic** — this skill is cosmetic. Don't restructure functions, change algorithms, or add features. Just strip the AI veneer. The only code changes should be:
   - Shortening variable names
   - Inlining trivial intermediates
   - Removing dead blank lines
   - Removing redundant else-after-return

5. **Report what you did** — after finishing, give a short summary: how many files touched, what kinds of things were removed. One or two sentences, not a bulleted essay.

## Example

Before:
```python
def calculate_model_accuracy(model_predictions, ground_truth_labels):
    """
    Calculate the accuracy of model predictions by comparing them
    against the ground truth labels.
    
    Args:
        model_predictions: The predictions made by the model
        ground_truth_labels: The true labels for comparison
    
    Returns:
        float: The calculated accuracy as a decimal value
    """
    # Initialize counter for correct predictions
    correct_prediction_count = 0
    
    # Iterate through all predictions and compare with ground truth
    for i in range(len(model_predictions)):
        # Check if prediction matches the ground truth
        if model_predictions[i] == ground_truth_labels[i]:
            # Increment the counter
            correct_prediction_count += 1
    
    # Calculate the final accuracy value
    calculated_accuracy = correct_prediction_count / len(ground_truth_labels)
    
    # Return the computed accuracy
    return calculated_accuracy
```

After:
```python
def accuracy(preds, labels):
    correct = sum(p == l for p, l in zip(preds, labels))
    return correct / len(labels)
```
