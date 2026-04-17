# Recommended Project Structure

Current state:
- This repository works well as a practical pipeline project.
- The next cleanup step is to separate `ui`, `pipeline`, `integrations`, `legacy`, and generated outputs.

Recommended tree:

```text
long2shorts/
├─ src/
│  └─ long2shorts/
│     ├─ ui/
│     │  └─ gui.py
│     ├─ pipeline/
│     │  ├─ analyze_longform.py
│     │  ├─ generate_shorts_packages.py
│     │  ├─ render_package_preview.py
│     │  └─ build_capcut_from_package.py
│     ├─ integrations/
│     │  ├─ openai_client.py
│     │  ├─ movie_info_lookup.py
│     │  └─ capcut/
│     │     ├─ draft_builder.py
│     │     └─ text_styles.py
│     ├─ models/
│     │  └─ package_schema.py
│     └─ paths.py
├─ docs/
│  ├─ PROJECT_STRUCTURE.md
│  └─ CAPCUT_STYLE_JSON.md
├─ templates/
│  └─ capcut_style_overrides.example.json
├─ analysis/
├─ generated_audio/
├─ legacy/
│  ├─ start_jiunsudaetong1_analysis.py
│  ├─ generate_jiunsudaetong1_shorts.py
│  └─ make_capcut_short.py
├─ long2shorts_gui.py
├─ analyze_longform.py
├─ generate_shorts_packages.py
├─ render_package_preview.py
└─ build_capcut_from_package.py
```

Suggested migration order:

1. Move one shared utility at a time into `src/long2shorts/`.
2. Move hard-coded one-off scripts into `legacy/`.
3. Keep the current root scripts as thin wrappers until everything is stable.
4. Move generated outputs and examples under `analysis/`, `generated_audio/`, `templates/`, and `docs/`.

Practical rule:
- Keep user-facing entrypoints stable now.
- Move internal logic first.
- Treat `analysis/` and `generated_audio/` as runtime data, not source code.
