# CapCut Style JSON

`build_capcut_from_package.py` now reads an optional `capcut` block from each package JSON.

Supported top-level sections:

- `capcut.draft`
  - `width`, `height`, `fps`
  - `video_volume`
  - `title_duration_sec`
- `capcut.title`
  - `font`
  - `style`
  - `border`
  - `background`
  - `clip`
  - `effect`
  - `bubble`
  - `material_patch`
  - `content_patch`
- `capcut.point`
  - same keys as `capcut.title`
- `capcut.narration`
  - `voice`
  - `speed`
  - `volume`

Per-caption override:

- Each item in `point_captions` may also include its own `capcut` block.
- The item block overrides `capcut.point`.

Useful fields:

```json
{
  "capcut": {
    "draft": {
      "video_volume": 0.85,
      "title_duration_sec": 3.8
    },
    "title": {
      "font": "Anton",
      "style": {
        "size": 14,
        "bold": true,
        "align": "center",
        "color": "#FFFFFF"
      },
      "border": {
        "color": "#111111",
        "width": 35
      },
      "clip": {
        "transform_y": -0.72
      }
    },
    "point": {
      "font": "Montserrat",
      "style": {
        "size": 10,
        "bold": true,
        "color": "#FFE96A"
      },
      "background": {
        "color": "#111111",
        "alpha": 0.45,
        "style": 1,
        "round_radius": 0.08,
        "height": 0.16,
        "width": 0.92
      },
      "clip": {
        "transform_y": 0.12
      },
      "material_patch": {
        "has_shadow": true,
        "shadow_alpha": 0.8,
        "shadow_color": "#000000",
        "shadow_distance": 5.0,
        "shadow_smoothing": 0.45
      }
    },
    "narration": {
      "voice": "onyx",
      "speed": 1.08,
      "volume": 1.0
    }
  }
}
```

Notes:

- `font` accepts CapCut font enum names such as `Anton`, `Montserrat`, `BebasNeue`, `Poppins_Bold`.
- `style.color` and `border.color` accept `#RRGGBB` or RGB arrays.
- `material_patch` merges into the outer CapCut text material JSON.
- `content_patch` merges into the decoded inner `content` JSON string for advanced cases.
- `effect` expects an `effect_id`.
- `bubble` expects both `effect_id` and `resource_id`.
