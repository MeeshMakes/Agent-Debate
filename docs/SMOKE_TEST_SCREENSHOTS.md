# Smoke Test Screenshots

These screenshots are captured from a live app run to confirm the main UI surfaces render correctly.

Capture script:

- `python scripts/capture_smoke_screenshots.py`

Generated images live in `docs/images/smoke/`.

---

## 1) Main Window Overview
![Main Window Overview](images/smoke/01-main-window-overview.png)

Shows the full desktop layout: center debate stream, left/right inner-monologue docks, bottom dock tabs, toolbar controls, and stats bar.

## 2) Center Live Debate Panel
![Center Live Debate Panel](images/smoke/02-center-live-debate-panel.png)

Primary audience view where public turns appear, with follow-text support and turn/speaker indicators.

## 3) Astra Inner Monologue Panel (Left)
![Astra Inner Monologue Panel](images/smoke/03-left-astra-inner-monologue-panel.png)

Dedicated private-thought feed for Astra used during think/speak cycles.

## 4) Nova Inner Monologue Panel (Right)
![Nova Inner Monologue Panel](images/smoke/04-right-nova-inner-monologue-panel.png)

Dedicated private-thought feed for Nova, parallel to Astra for side-by-side reasoning inspection.

## 5) Arbiter Panel
![Arbiter Panel](images/smoke/05-arbiter-panel.png)

Intervention surface for moderation, capture controls, and injected guidance flow.

## 6) Debate Graph Panel
![Debate Graph Panel](images/smoke/06-debate-graph-panel.png)

Branch/sub-topic graph surface for tracking exploration structure over time.

## 7) Scoring & Verdict Panel
![Scoring and Verdict Panel](images/smoke/07-scoring-verdict-panel.png)

Session-level scoring timeline and verdict summary area used for post-round assessment.

## 8) Debate Studio (Topic Picker)
![Debate Studio Topic Picker](images/smoke/08-debate-studio-topic-picker.png)

Topic/debate setup workspace: custom debate editing, repo watchdog controls, session/transcript tooling, and assistant panel.

## 9) Session Browser Dialog
![Session Browser Dialog](images/smoke/09-session-browser-dialog.png)

Past-session explorer with replay/delete actions and transcript preview.

## 10) Analytics Dialog
![Analytics Dialog](images/smoke/10-analytics-dialog.png)

Cross-session analytics dashboard showing aggregate counts, win rates, and per-session details.
