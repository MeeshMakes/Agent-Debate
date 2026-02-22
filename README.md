# Agent Debate NextGen

An interactive desktop app where two AI agents debate topics in real time while you watch, steer, and inspect their reasoning.

This is built for people who want **serious AI conversations** without sending data to the cloud by default.

---

## Why people use this

- **Watch ideas clash live** in a clean, center-stage debate stream.
- **See reasoning, not just answers** with private thought channels per agent.
- **Intervene like a moderator** using an arbiter panel and live injections.
- **Debate your own topics** with custom briefs and talking points.
- **Link a real code repository** and let debates adapt as files change.
- **Keep it local-first** with Ollama-based model runtime.

---

## What the experience feels like

Think of it as a live show + analysis lab:

- The middle panel shows the public debate, turn by turn.
- Side panels reveal each agent’s private internal reasoning.
- A scoring and graph view tracks momentum, branches, and outcomes.
- You can pause, inject guidance, continue, and compare how the debate evolves.

It’s designed to feel understandable and visual even if you’re not a programmer.

---

## Best use cases

- Breaking down complex ideas (science, philosophy, policy, faith, ethics)
- Stress-testing your own arguments before presentations or writing
- Exploring “both sides” of a controversial topic without echo chambers
- Reviewing software architecture with AI agents that can debate tradeoffs
- Running repository-linked debates for ongoing system understanding

---

## Key features (consumer-level)

### 1) Debate Studio
- Pick a starter topic or create your own custom debate.
- Add context and talking points in plain language.

### 2) Live Dual-Agent Debate
- Two agents argue, challenge, and refine positions over multiple turns.
- An arbiter can summarize progress and unresolved tension.

### 3) “Follow Text” narration mode
- Text-to-speech can read the debate stream out loud.
- Live highlight tracks spoken words for easy follow-along.

### 4) Knowledge ingestion
- Add files/folders so debates can use your material as context.
- Useful for research notes, documents, and codebases.

### 5) Repo-linked debates (Watchdog mode)
- Link a repository to a debate.
- Build a semantic dataset of the repo for broader system understanding.
- Detect file changes and refresh context during active debate sessions.

### 6) Session memory and history
- Browse prior sessions, transcripts, and evolution over time.
- Re-open and continue debates instead of starting from scratch.

---

## Privacy and runtime model

- Built for **local-first usage**.
- Uses Ollama-compatible model endpoints.
- No OpenAI API dependency is required for normal runtime.

---

## Quick start

### Requirements
- Python 3.11+
- Ollama running locally (or reachable endpoint)

### Launch
1. Install dependencies:

   `pip install -r requirements.txt`

2. Start the app:

   `python -m app.main`

---

## First 5 minutes (recommended)

1. Open Debate Studio.
2. Select a starter topic or create a Custom Debate.
3. Press Start and watch the first round.
4. Pause and inject one arbiter instruction.
5. Continue and observe how agent behavior changes.

For repository analysis:

1. Create a Custom Debate in Repo Watchdog mode.
2. Link your repository.
3. Run “Analyze Repo & Fill Debate Schema.”
4. Start debate and iterate as changes are detected.

---

## For power users

If you want model packs, tuning details, and runtime notes, see:

- `LOCAL_MODELS.md`
- `docs/SMOKE_TEST_SCREENSHOTS.md` (UI smoke-test screenshot gallery)

---

## UI screenshots (on-main gallery)

### 1) Main Window Overview
![Main Window Overview](docs/images/smoke/01-main-window-overview.png)

Shows the full desktop layout: center debate stream, left/right inner-monologue docks, bottom dock tabs, toolbar controls, and stats bar.

### 2) Center Live Debate Panel
![Center Live Debate Panel](docs/images/smoke/02-center-live-debate-panel.png)

Primary audience view where public turns appear, with follow-text support and turn/speaker indicators.

### 3) Astra Inner Monologue Panel (Left)
![Astra Inner Monologue Panel](docs/images/smoke/03-left-astra-inner-monologue-panel.png)

Dedicated private-thought feed for Astra used during think/speak cycles.

### 4) Nova Inner Monologue Panel (Right)
![Nova Inner Monologue Panel](docs/images/smoke/04-right-nova-inner-monologue-panel.png)

Dedicated private-thought feed for Nova, parallel to Astra for side-by-side reasoning inspection.

### 5) Arbiter Panel
![Arbiter Panel](docs/images/smoke/05-arbiter-panel.png)

Intervention surface for moderation, capture controls, and injected guidance flow.

### 6) Debate Graph Panel
![Debate Graph Panel](docs/images/smoke/06-debate-graph-panel.png)

Branch/sub-topic graph surface for tracking exploration structure over time.

### 7) Scoring & Verdict Panel
![Scoring and Verdict Panel](docs/images/smoke/07-scoring-verdict-panel.png)

Session-level scoring timeline and verdict summary area used for post-round assessment.

### 8) Debate Studio (Topic Picker)
![Debate Studio Topic Picker](docs/images/smoke/08-debate-studio-topic-picker.png)

Topic/debate setup workspace: custom debate editing, repo watchdog controls, session/transcript tooling, and assistant panel.

### 9) Session Browser Dialog
![Session Browser Dialog](docs/images/smoke/09-session-browser-dialog.png)

Past-session explorer with replay/delete actions and transcript preview.

### 10) Analytics Dialog
![Analytics Dialog](docs/images/smoke/10-analytics-dialog.png)

Cross-session analytics dashboard showing aggregate counts, win rates, and per-session details.

---

## Status

Actively evolving project with rapid iteration on:

- Debate quality
- Repo understanding workflows
- UI/UX polish
- Follow-mode and transcript experience

If you’re testing builds frequently, expect fast improvements between versions.
