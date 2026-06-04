# CapCut Style JSON

`build_capcut_from_package.py` reads an optional `capcut` block from each package JSON.

The current default style is tuned for `short_templet.png` and the reference short:
`https://www.youtube.com/shorts/3ehCTHyCm6U`.

## Main Sections

- `capcut.draft`: `width`, `height`, `fps`, `video_volume`, `title_duration_sec`, `title_full_duration`
- `capcut.template`: `enabled`, `image_path`, `clip`
- `capcut.text_overlay`: `enabled`
- `capcut.title`: rendered top title settings
- `capcut.channel`: rendered bottom source channel settings
- `capcut.point`: regular point-caption settings
- `capcut.narration`: `voice`, `speed`, `volume`

`capcut.title` uses two separate rendered lines:

- `line1`: black text to the right of the top-left channel mark
- `line2`: green text with a black outline, centered under line 1

`capcut.channel` is centered in the lower source area. The template already contains the heart icon, so the default source text is just the channel name.

## Example

```json
{
  "capcut": {
    "draft": {
      "video_volume": 0.85,
      "title_duration_sec": 3.8,
      "title_full_duration": true
    },
    "template": {
      "enabled": true,
      "image_path": "short_templet.png",
      "clip": {
        "scale_x": 1.0,
        "scale_y": 1.0
      }
    },
    "text_overlay": {
      "enabled": true
    },
    "title": {
      "font_path": "assets/fonts/Jua-Regular.ttf",
      "line1": {
        "box": [290, 70, 1040, 165],
        "max_chars": 12,
        "image_font_size": 78,
        "min_image_font_size": 58,
        "image_color": "#111111",
        "image_align": "left"
      },
      "line2": {
        "box": [80, 245, 1000, 380],
        "max_chars": 11,
        "image_font_size": 76,
        "min_image_font_size": 56,
        "image_color": "#00E846",
        "image_stroke_color": "#111111",
        "image_stroke_width": 9,
        "image_align": "center"
      }
    },
    "channel": {
      "font_path": "assets/fonts/Jua-Regular.ttf",
      "suffix": "",
      "box": [0, 1515, 1080, 1610],
      "image_font_size": 72,
      "min_image_font_size": 52,
      "image_color": "#111111",
      "image_align": "center"
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

## Notes

- `font_path` is used by rendered text overlays. The bundled default is `assets/fonts/Jua-Regular.ttf`.
- Rendered title/source overlays default to Jua. If `capcut.text_overlay.enabled` is false, editable CapCut text falls back to `Poppins_Bold` for the title and `Montserrat` for point captions/source text.
- `style.color`, `border.color`, `image_color`, and `image_stroke_color` accept `#RRGGBB`.
- `short_templet.png` has a transparent video window from y=432 to y=1487.
- You can override the frame image at runtime with `--template-image path/to/frame.png`.
- `--channel-name` overrides the auto-detected bottom source label. By default that is the YouTube channel name, or the movie title for local/movie sources.
- Each item in `point_captions` may include its own `capcut` block; that item block overrides `capcut.point`.
- `material_patch` merges into the outer CapCut text material JSON.
- `content_patch` merges into the decoded inner `content` JSON string for advanced cases.
- `effect` expects an `effect_id`.
- `bubble` expects both `effect_id` and `resource_id`.
