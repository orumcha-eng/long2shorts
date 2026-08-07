# Long2Shorts

긴 영상이나 YouTube 영상을 분석해서 쇼츠 후보를 뽑고, 미리보기 영상과 CapCut 초안 패키지까지 만드는 로컬 파이프라인입니다.

## 빠른 시작

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

`.env`에는 최소 `OPENAI_API_KEY`를 넣으세요. YouTube 댓글/메타데이터까지 수집하려면 `YOUTUBE_API_KEY` 또는 `GOOGLE_API_KEY`도 넣습니다.

예전 작업 환경을 위해 `..\auto_Youtube\shorts\.env`도 계속 읽지만, 이 프로젝트 안의 `.env`가 우선입니다.

## GUI 실행

```powershell
.\.venv\Scripts\python.exe long2shorts_gui.py
```

주요 흐름:

1. 자동 운영은 `트렌드 자동 생성`을 누릅니다.
2. 앱이 최근 72시간 예능/연예 롱폼 후보를 찾고, green 후보 1개를 자동 선택합니다.
3. 선택된 원본을 다운로드하고, 자막/STT와 댓글 반응을 바탕으로 쇼츠 패키지를 생성합니다.
4. 품질 기준을 통과한 쇼츠 패키지마다 CapCut 초안을 자동 생성합니다.
5. CapCut에서 최종 컷, 자막, 음악만 마무리합니다.

수동 운영:

1. YouTube 링크를 넣고 `다운로드+분석`을 실행합니다.
2. 후보가 생성되면 목록에서 쇼츠 패키지를 선택합니다.
3. `미리보기 열기`로 러프컷을 확인합니다.
4. `선택 항목 CapCut 만들기`로 CapCut 초안을 생성합니다.

## CLI 흐름

YouTube 링크 하나로 댓글/메타데이터 수집, 영상 다운로드, 자막 또는 오디오 전사, 패키지 생성을 이어서 실행:

```powershell
.\.venv\Scripts\python.exe collect_youtube_context.py --url "https://www.youtube.com/watch?v=VIDEO_ID" --download-video --prepare-transcript --generate-packages
```

최근 트렌드 롱폼 후보 탐색:

```powershell
.\.venv\Scripts\python.exe discover_trending_sources.py --window-hours 72 --limit 20
```

결과는 `analysis\trends\latest_trend_candidates.json`에 저장됩니다. 한 번 자동 처리한 원본은 `analysis\trends\trend_history.json`에 기록되어 다음 실행에서 제외됩니다.

로컬 영상 분석:

```powershell
.\.venv\Scripts\python.exe analyze_longform.py --source-video "input.mp4"
.\.venv\Scripts\python.exe generate_shorts_packages.py --analysis-dir "analysis\input" --source-title "영상 제목"
```

미리보기 렌더링:

```powershell
.\.venv\Scripts\python.exe render_package_preview.py --source-video "input.mp4" --package "analysis\input\shorts_candidates\final\short_01.json"
```

CapCut 초안 생성:

```powershell
.\.venv\Scripts\python.exe build_capcut_from_package.py --source-video "input.mp4" --package "analysis\input\shorts_candidates\final\short_01.json"
```

## 산출물 위치

- `analysis/`: 전사 결과, 댓글 컨텍스트, 쇼츠 후보, 미리보기
- `downloads/`: YouTube 다운로드 파일
- `generated_audio/`: TTS 내레이션 캐시
- `generated_text_overlays/`: 제목/출처 텍스트 오버레이 이미지

대용량 영상과 생성 산출물은 `.gitignore`에 포함되어 있습니다.

## 참고 문서

- `docs/PROJECT_STRUCTURE.md`: 다음 구조 정리 방향
- `docs/CAPCUT_STYLE_JSON.md`: CapCut 스타일 JSON 옵션
- `docs/HOME_PC_SETUP.md`: 집 PC·새 OpenAI/YouTube 계정 이전 및 첫 실행 절차
