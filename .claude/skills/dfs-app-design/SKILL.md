---
name: dfs-app-design
description: UX/UI design system and product patterns for the DFS-NFL subscription app. Use when designing, mocking up, or building any user-facing surface for this product — screens, flows, components, landing pages, or emails.
---

# DFS-NFL App Design System

You are designing surfaces for a mobile-first DFS projections + lineup
subscription product built on this repo's data (see README.md and
`outputs/projections/` for real column semantics). Follow this document;
where it is silent, fall back on strong mobile-first defaults. For visual
mockups, invoke the built-in `design` skill (Claude Design canvas) and
apply these tokens there.

## Product principles

1. **Decisions, not data.** Users open the app to answer: who do I play,
   who do I fade, is my lineup good? Every screen leads to an action.
   Raw stats are one tap deeper, never the first thing shown.
2. **Sunday-morning thumb test.** Peak usage is 15 minutes before lock,
   one-handed, outdoors. Primary actions in thumb reach, tap targets
   ≥44px, no hover-dependent anything, instant loads (projections are
   static per run — cache aggressively, render instantly).
3. **Uncertainty is the product.** We sell calibrated ranges, not fake
   precision. Never show a projection without its p10–p90 context.
   One decimal place max. No 20.023529052734375, ever.
4. **Honest track record.** The accuracy log is a first-class screen,
   losses included. Trust is the moat for a paid picks product.

## Signature components

- **Player card:** name/team/opp/salary on line 1; projection with a
  horizontal p10–p90 **range bar** (p50 tick, projection dot) as the
  hero element; value (pts/$1K) and a volatility chip. Tap → detail
  sheet with trailing games, Vegas context, skew explanation.
- **Volatility chip:** `STEADY` (cv < 0.45), `SWING` (0.45–0.7),
  `BOOM/BUST` (> 0.7, or skew > 0.25). Always paired with an icon or
  label, never color alone.
- **Mode toggle:** global **Cash / GPP** switch that re-ranks every list
  (Cash sorts floor-weighted: p10 and low cv; GPP sorts ceiling-weighted:
  p90, value_p90, positive skew). This toggle is the app's identity —
  persistent, top-of-screen, animated on switch.
- **Lineup builder:** bottom-sheet salary tracker (remaining $ and
  $/slot-remaining, turns amber < $4K/slot), roster slots as chips,
  projected total with a stacked p10/p90 band for the whole lineup.
- **Confidence tag:** cold-start (`low`) players show a "rookie/limited
  data" tag — never present them with the same certainty as veterans.
- **Injury handling:** Q = amber dot beside name; O/IR/D = row greyed,
  excluded from optimizer by default with a visible override.

## Tokens

Color (dark-first — evening/game-time usage, but ship both themes):
- Ground dark `#101614` / light `#f6f8f6` (turf-biased neutrals)
- Surface dark `#1a221e` / light `#ffffff`
- Ink dark `#e8ede9` / light `#1c2420`; muted `#93a29a` / `#5d6a62`
- Accent (actions, active mode): turf green `#34c07c` dark / `#22754e` light
- Ceiling/GPP accent: amber `#e0a33a` dark / `#9a6b14` light
- Semantic: positive `#34c07c`, warning `#e0a33a`, negative `#e05c4f`
  (semantic ≠ accent duty; don't reuse accent for "good stat")
- Range bars: muted track, accent-filled p10–p90 span

Type:
- Display/headings: a condensed athletic grotesque (Saira Condensed or
  Barlow Condensed), uppercase for section headers only
- Body/UI: Source Sans 3 or system stack
- **All numbers: tabular figures** (`font-variant-numeric: tabular-nums`),
  mono acceptable for salary/points columns

Spacing/shape: 4px grid; cards 12px radius; chips fully rounded; one
shadow level only.

## Screens (v1 scope)

1. **Slate home** — slate picker, countdown to lock, top values, news-ish
   deltas ("projection risers since Tuesday")
2. **Players** — filterable ranked list (position tabs, Cash/GPP aware),
   the player card grid
3. **Optimizer** — constraints in (locks, excludes, exposure), lineups
   out, each with total projection band and salary used
4. **My lineups** — saved lineups with live projected totals
5. **Track record** — weekly scorecard vs actuals, calibration chart
6. **Paywall/account** — free tier: top-10 values + 1 lineup; paid:
   everything. Show real (delayed) data in previews, not lorem.

## Voice

Plain, confident, numerate; explain like a sharp friend, not a tout.
Say "range," "floor," "ceiling" — not "sigma," "quantile," "cv" (those
live in tooltips). Never promise profit; the track record speaks.
