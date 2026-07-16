# Daily Shorts Orchestrator

`오늘 쇼츠 제작`은 매번 같은 프롬프트를 반복하는 명령이 아니다. 실행 전 업로드 성과를 읽고, 다음 실험 규칙을 결정한 뒤, 그 규칙을 패키지 생성과 렌더에 강제로 전달한다.

## Source selection

By default, `source_acquisition.enabled` is `true`. A daily run researches recent trend candidates, takes the selected green longform, downloads it, prepares transcript/context, and treats that prepared source as today's production input.

You can still add a fixed permitted library source when you want the daily run to choose from your own pool as well:

```powershell
.venv\Scripts\python.exe shorts_orchestrator.py --config automation_config.json register-source --source-video "D:\media\my_longform.mp4" --analysis-dir "analysis\my_longform" --source-title "My Longform" --confirm-owned
```

## Daily run

Use the `오늘 쇼츠 제작` button, or run:

```powershell
.venv\Scripts\python.exe shorts_orchestrator.py --config automation_config.json daily-run --execute
```

Each run performs these steps in order:

1. Collect due 1h, 24h, 72h, and 7d performance snapshots.
2. Convert the evidence into one active learning rule for this run.
3. Research current trend signals and select the best eligible longform.
4. Download the selected longform, prepare transcript/comment context, and add it to the run's source list.
5. Generate scored Shorts packages with the active learning rule.
6. Render the highest scoring `auto_render` package as a 1080x1920 MP4 with Pretendard title text and point captions.

The rendered output is saved under the selected source's `analysis/.../productions` directory. The default is one rendered candidate per source so there is a deliberate final review point.

## Review and approval

Every rendered MP4 is automatically registered as a review item with a lightweight QA result. The daily run does not ask you to manage files by hand; you only inspect the rendered video and record the decision.

```powershell
.venv\Scripts\python.exe shorts_orchestrator.py --config automation_config.json list-reviews
.venv\Scripts\python.exe shorts_orchestrator.py --config automation_config.json review-decision --output "analysis\my_longform\productions\short_01.mp4" --status approved --note "ready"
```

Supported decisions are `approved`, `revision_requested`, and `rejected`. If YouTube upload is enabled, `require_review_approval` defaults to `true`, so approved renders can be uploaded with:

```powershell
.venv\Scripts\python.exe shorts_orchestrator.py --config automation_config.json upload-approved --execute
```

## Learning loop

After at least three evaluated uploads, the next run changes its required edit rule from observed retention rather than reusing the prior run blindly. For example, low retention first switches to a reaction/payoff-first hook; if that remains weak, it switches to a visual-question-first hook. The active rule and its reason are saved in `analysis/automation/runs/<run_id>/learning_rule.json` and embedded in the generated package JSON.

## YouTube Analytics setup

Set `youtube_analytics.enabled` to `true`, place the OAuth client JSON at the configured secret path, then authorize once:

```powershell
.venv\Scripts\python.exe shorts_orchestrator.py --config automation_config.json authorize-youtube
```

The token is stored under `secrets/`, which is ignored by Git. Register each published video with `register-published` so the scheduled checkpoints can feed the following daily run.

## Optional private upload

After the OAuth setup, set `youtube_upload.enabled` to `true`. The default `privacy_status` is `private`, so a daily run can upload the rendered MP4 for final YouTube review without making it public. The orchestrator remembers the SHA-256 of every uploaded render and will not upload the same file twice.

Set `privacy_status` to `public` only when the channel is ready for unattended publication. Public uploads are automatically registered for the next analytics checkpoints.
