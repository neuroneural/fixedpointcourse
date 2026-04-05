# Project rules

This repo contains lecture slides built with reveal.js.

## Goals
- Preserve existing slide structure unless explicitly asked to reorganize it.
- Prefer small, local edits over large rewrites.
- Keep slides readable in presentation mode.

## reveal.js conventions
- Use standard reveal.js sections for slides.
- Do not introduce frameworks or build tools unless already present.
- Prefer speaker notes only when explicitly useful.
- Keep per-slide text concise.
- Avoid overcrowding slides; split dense material into multiple slides.
- Preserve fragment behavior, transitions, and custom plugins unless asked to change them.

## Styling
- Reuse existing CSS classes and theme choices.
- Do not add inline styles unless necessary.
- Keep code examples minimal and presentation-friendly.

## Workflow
- Before large edits, inspect the current slide structure and theme files.
- After changes, check for broken HTML/JS and obvious reveal.js issues.
- When adding math, use the project’s existing math setup.
- When adding diagrams, prefer SVG or simple HTML/CSS over heavy dependencies.

## Output style
- When proposing changes, explain which slides were changed and why.
- When uncertain, preserve existing wording and only fix correctness, clarity, or layout.
