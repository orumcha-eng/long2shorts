# 집 PC 이전·새 계정 설정

이 문서는 다른 Windows PC와 다른 OpenAI·Google 계정에서도 현재 쇼츠 제작 흐름을 안전하게 이어가기 위한 절차다. Git에는 코드와 예시 설정만 올리고, API 키·YouTube OAuth 파일·Telegram 토큰·분석 이력은 올리지 않는다.

## 1. 코드 받기와 실행 환경

PowerShell에서 실행한다.

```powershell
git clone https://github.com/orumcha-eng/long2shorts.git
Set-Location long2shorts
git checkout codex/shorts-automation-studio
git pull
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
Copy-Item templates\automation_config.example.json automation_config.json
```

이미 저장소가 있다면 `git pull`만 실행하고, 의존성이 바뀐 경우에만 `pip install -r requirements.txt`를 다시 실행한다.

## 2. 새 OpenAI 계정

집 PC의 `.env`에 **새 계정에서 발급한** API 키를 넣는다. 키는 Git이나 채팅에 올리지 않는다.

```dotenv
OPENAI_API_KEY=새_OpenAI_API_키
YOUTUBE_API_KEY=YouTube_Data_API_키
```

`YOUTUBE_API_KEY`는 트렌드·영상 메타데이터·댓글 수집용이다. `GOOGLE_API_KEY`를 사용 중이면 그 이름으로 대신 넣어도 된다.

## 3. 새 YouTube 채널 연결

새 채널을 올릴 Google 계정으로 Google Cloud Console에서 OAuth 2.0 **데스크톱 앱** 클라이언트를 만든다. 해당 프로젝트에서 다음 API를 활성화한다.

- YouTube Data API v3
- YouTube Analytics API

다운로드한 OAuth JSON을 다음 이름으로 둔다.

```powershell
New-Item -ItemType Directory -Force secrets | Out-Null
Copy-Item "다운로드한_클라이언트_JSON_경로" secrets\youtube_oauth_client.json
```

그다음 브라우저 인증을 한 번 실행한다. 브라우저에서는 **집 PC에서 사용할 새 YouTube 채널 계정**으로 로그인해야 한다.

```powershell
.\.venv\Scripts\python.exe shorts_orchestrator.py authorize-youtube
.\.venv\Scripts\python.exe shorts_orchestrator.py youtube-status
```

정상이라면 `youtube-status`가 새 채널명을 출력한다. 토큰은 `secrets\youtube_analytics_token.json`에 저장된다. 두 파일 모두 Git에서 제외되어 있다.

## 4. 예약 업로드 정책 확인

`automation_config.json`에서 다음 값을 유지한다.

```json
"youtube_upload": {
  "enabled": true,
  "require_review_approval": true,
  "privacy_status": "public",
  "publish_schedule": {
    "enabled": true,
    "mode": "daily_slots",
    "daily_slots": ["09:00", "12:00", "15:00", "18:00", "21:00"]
  }
}
```

승인된 여러 영상은 새 채널의 마지막 게시·예약 시점 다음 슬롯부터 순서대로 예약된다. 21시 이후에는 다음 날 9시부터 시작한다. 승인 전에는 업로드되지 않는다.

## 5. Telegram 새 봇 설정(선택)

새 Telegram 계정이나 새 봇을 쓸 경우 `@BotFather`에서 발급한 토큰과 본인의 숫자 chat ID를 `.env`에 넣는다.

```dotenv
TELEGRAM_BOT_TOKEN=새_봇_토큰
TELEGRAM_ALLOWED_CHAT_IDS=본인의_숫자_chat_ID
```

처음에는 아래처럼 직접 실행하고 `/start`, `/id`를 보낸다. chat ID를 넣은 뒤 재시작한다.

```powershell
.\Start-TelegramBot.ps1
```

자동 시작은 한 번만 설치한다.

```powershell
.\Install-TelegramBotStartup.ps1
```

## 6. 첫 제작과 업로드 확인

먼저 제작만 한다. 새 원본을 최근 3일 반응 기준으로 탐색하고, 방송사·런닝맨 차단 규칙과 20초 초과·화면 품질 기준을 적용한다.

```powershell
.\.venv\Scripts\python.exe shorts_orchestrator.py daily-run --execute
.\.venv\Scripts\python.exe shorts_orchestrator.py list-reviews --status needs_review
```

Telegram 카드 또는 GUI에서 확인한 뒤에만 승인한다. CLI를 사용할 경우에는 **정확한 출력 파일 경로**를 지정한다.

```powershell
.\.venv\Scripts\python.exe shorts_orchestrator.py review-decision --output "analysis\...\short_01.mp4" --status approved --note "ready"
.\.venv\Scripts\python.exe shorts_orchestrator.py upload-approved --execute --limit 1
```

업로드 전 마지막 연결 점검은 다음 명령으로 한다.

```powershell
.\.venv\Scripts\python.exe shorts_orchestrator.py youtube-status --sync-metrics
```

## 주의

- `analysis/`, `downloads/`와 `secrets/`는 PC별 작업 이력·영상·인증 정보다. Git으로 옮기지 않는다.
- 기존 PC의 OAuth 토큰을 복사하지 말고, 새 채널 계정으로 다시 인증한다.
- 새 OpenAI 키는 집 PC의 `.env`에만 넣는다.
- `shopping_shorts/`는 쇼츠 자동화와 별개 프로젝트이므로 이 절차에 포함하지 않는다.
