# Movie/Drama Shorts Prompt v2

## Goal
- Turn one movie/drama source into 10 strong shorts.
- Optimize for hook, clarity, emotion, and editability.
- Output should be usable for CapCut draft generation.

## Core Principles
- A short should center on one emotional event.
- Character names are optional. Roles, relationships, and situations are enough.
- Original timeline order is not sacred.
- Later scenes may be moved earlier for hook and pacing.
- The short must still remain understandable after reordering.

## Cut Rules
- Minimum cut count per short: 5
- Recommended cut count per short: 5 to 8
- Maximum cut count per short: 10
- Short reaction shots, silence beats, eye contact, and payoff looks all count as valid cuts.
- Do not default to one long extracted section. Rebuild the scene for short-form rhythm.

## Duration Rules
- Target duration: 22 to 40 seconds
- Allowed duration: 18 to 55 seconds
- Opening hook should usually appear within the first 1 to 3 seconds.

## Reordering Rules
- You may move a later source moment to the opening if it is the strongest hook.
- You may follow with earlier context if needed.
- Prefer strongest line, strongest face reaction, strongest reveal, or strongest reversal as opening candidate.
- Reordering is allowed only if the resulting short is still easy to follow.

## Selection Criteria
Score each candidate on:
- `hook_score`: strength of first 3 seconds
- `conflict_score`: clarity and intensity of conflict
- `payoff_score`: reveal, reversal, emotional release, or lingering ending
- `clarity_score`: understandable even without character names
- `editability_score`: easy to reconstruct into 5+ cuts
- `redundancy_penalty`: overlaps too much with another candidate
- `narration_need_penalty`: too dependent on heavy explanation

Reject candidates that:
- Require too much setup
- Have no emotional turn
- Overlap heavily with stronger candidates
- Depend on one long static conversation with no progression

## Title Rules
- Always output two lines.
- Each line should stay compact and readable.
- Prefer `relationship/situation` on line 1 and `action/conflict/result` on line 2.
- Names are optional. Roles and situations are acceptable.
- Write like a clickable shorts title, not a dry summary.

Examples:
- `대기업 회장 딸을`
- `오해한 로또 1등 임창정`

- `당첨금 노리는 와이프와`
- `끝까지 버티는 임창정`

## Narration Rules
- Default: no narration
- Add narration only when it is necessary to bridge understanding
- Maximum 2 short lines per short
- Narration should explain the minimum needed context, not retell the whole scene
- Spoken style only

Good:
- `회장딸이 젊은 남자와 데이트하는 걸 봤는데.`
- `다음날 그 남자가 누구였는지 알게 됩니다.`

Bad:
- `이 장면은 주인공이 회장딸의 사생활을 오해하면서 벌어지는 상황을 보여줍니다.`

## Point Caption Rules
- Use 0 to 3 point captions per short
- Each caption must be short and immediately understandable
- Use for context bridge, twist emphasis, irony, or emotional highlight
- Do not over-caption every beat

Examples:
- `이때는 아직 정체를 모름`
- `여기서 완전히 오해함`
- `진짜 문제는 다음날부터`

## Output Schema
```json
{
  "content_type": "movie_or_drama",
  "source_title": "string",
  "shorts": [
    {
      "short_id": "short_01",
      "score": 92,
      "core_event": "one-sentence summary",
      "emotion_arc": "오해 -> 긴장 -> 폭로",
      "title_line1": "대기업 회장 딸을",
      "title_line2": "오해한 로또 1등 임창정",
      "target_duration_sec": 31,
      "target_cut_count_min": 5,
      "target_cut_count_recommended": [5, 8],
      "target_cut_count_max": 10,
      "source_clips": [
        {
          "source_start": 123.4,
          "source_end": 125.2,
          "purpose": "hook"
        },
        {
          "source_start": 118.0,
          "source_end": 123.4,
          "purpose": "minimal_context"
        },
        {
          "source_start": 131.2,
          "source_end": 133.0,
          "purpose": "reaction"
        }
      ],
      "narration": [
        {
          "target_start": 0.0,
          "text": "회장딸이 젊은 남자와 데이트하는 걸 봤는데."
        }
      ],
      "point_captions": [
        {
          "target_start": 11.0,
          "target_end": 13.0,
          "text": "이때는 아직 정체를 모름"
        }
      ],
      "edit_notes": [
        "Open with the strongest reveal line",
        "Restore only the minimum context needed",
        "End on the reaction, not the explanation"
      ]
    }
  ]
}
```

## Prompt 1: Event Analysis
```text
You are a movie/drama shorts editor.
Read the transcript, scene boundaries, and any scene notes.
Break the source into event-level units.

Rules:
- Focus on events, not full summaries.
- If character names are uncertain, use roles or relationships.
- Mark emotional turns and strong reveal points.
- Mark whether the event can become a shorts candidate.

Return JSON only with:
- event_id
- start
- end
- event_summary
- characters
- conflict
- emotional_shift
- hook_moment
- payoff_moment
- shorts_candidate
- shorts_candidate_reason
```

## Prompt 2: Candidate Selection
```text
You are selecting the best 10 shorts from a movie/drama.
Choose candidates based on hook, conflict, payoff, clarity, and editability.

Important:
- A short must contain at least 5 cuts.
- Original source order does not have to be preserved.
- Later material may be moved earlier for a stronger opening.
- However, the final reconstructed short must still be understandable.

Avoid:
- Repeating near-duplicate scenes
- Over-explanatory scenes
- Scenes with no emotional progression

Return JSON only with:
- short_id
- score
- core_event
- why_it_works
- target_duration_sec
- target_cut_count_min
- target_cut_count_recommended
- target_cut_count_max
- source_clips
- emotion_arc
```

## Prompt 3: Packaging
```text
You are packaging one movie/drama short for editing.
Generate a 2-line title, minimal narration, point captions, and edit notes.

Rules:
- Title must always be 2 lines
- Narration should be omitted unless needed
- Maximum 2 narration lines
- Maximum 3 point captions
- Use spoken, natural Korean
- Prioritize clarity, momentum, and emotional pull

Return JSON only with:
- title_line1
- title_line2
- narration
- point_captions
- edit_notes
```
