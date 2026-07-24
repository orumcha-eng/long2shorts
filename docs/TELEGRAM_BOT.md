# Telegram control bot

This optional bot runs on this PC and receives Telegram commands through long polling. It needs no public IP address, but the PC must be powered on and connected to the internet.

## 1. Create a private bot

1. In Telegram, open `@BotFather` and use `/newbot`.
2. Copy its bot token. Treat this token like a password; do not paste it into source code or chat.
3. Add the following to the project's `.env` file:

```dotenv
TELEGRAM_BOT_TOKEN=replace-with-your-bot-token
TELEGRAM_ALLOWED_CHAT_IDS=
```

## 2. Get and allow your chat ID

Start the bot manually once:

```powershell
.\Start-TelegramBot.ps1
```

Send `/start` to your new bot. While no allowed ID is configured, it replies only with your chat ID and a setup instruction. Put that number into `.env`, then stop the script with `Ctrl+C` and restart it:

```dotenv
TELEGRAM_ALLOWED_CHAT_IDS=123456789
```

Multiple people can be allowed with comma-separated numeric IDs. Do not leave this value blank after setup.

## 3. Check and run

```powershell
.\.venv\Scripts\python.exe telegram_bot.py --check
.\Start-TelegramBot.ps1
```

Available private commands:

- `/status` — persisted production status
- `/reviews` — rendered Shorts waiting for review
- `/approve short_01` — approve a rendered Short
- `/revise short_01 feedback` — request revision with a note
- `/create` — start the normal daily Shorts production flow
- `/upload` then `/upload confirm` — upload approved videos and apply the configured YouTube schedule
- `/logs` and `/stop` — inspect or stop a Telegram-started job

The upload command deliberately needs the second `confirm` message. The bot never changes existing YouTube reservations.

## 4. Start automatically after Windows sign-in

After the manual run works, open PowerShell in this project and run once:

```powershell
.\Install-TelegramBotStartup.ps1
```

It creates a Windows Scheduled Task called `Long2Shorts Telegram Bot` that starts hidden each time you sign in. On PCs where Scheduled Tasks are restricted, it automatically creates an equivalent launcher in your personal Startup folder. To remove it later:

```powershell
Unregister-ScheduledTask -TaskName "Long2Shorts Telegram Bot" -Confirm:$false
Remove-Item "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Startup\Long2Shorts Telegram Bot.vbs" -ErrorAction SilentlyContinue
```
