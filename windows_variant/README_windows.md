# Установка на Windows-сервер

Развёртывание на чистом Windows Server (или Windows 10/11). Код приложения тот же,
что и для Linux/Mac; отличаются установка зависимостей (PowerShell + winget),
запуск Qdrant (Docker Desktop) и автозапуск (Scheduled Task).

## Требования

- Windows Server 2022/2025 или Windows 10/11 с **winget** («App Installer» из
  Microsoft Store обычно уже есть; на Server core может потребоваться установить).
- Права администратора (PowerShell «Запуск от имени администратора»).
- Для Qdrant — **Docker Desktop** (нужны WSL2/Hyper-V и перезагрузка). Без него
  приложение запустится, но поиск работать не будет до запуска Qdrant.
- Для генерации — **Ollama для Windows** (ставится автоматически). GPU NVIDIA
  опционально (ключ `-Cuda`).

## Установка одной командой (локально)

PowerShell от администратора, из папки проекта:

```powershell
powershell -ExecutionPolicy Bypass -File windows_variant\setup_windows.ps1 `
    -AdminToken "придумайте-пароль" -Model "qwen2.5:14b-instruct"
```

Скрипт: ставит Python/Git/Ollama (winget), качает модель, поднимает Qdrant в
Docker (если установлен), создаёт `.venv` и зависимости, прописывает `.env`
(папка документов по умолчанию `C:\rag\db`, бэкенд Ollama, Whisper = faster) и
регистрирует автозапуск (Scheduled Task `RagApi`). Откройте `http://localhost:8000`.

С GPU NVIDIA добавьте `-Cuda` (поставит torch под CUDA `cu124`).

## Деплой из GitHub

```powershell
powershell -ExecutionPolicy Bypass -File windows_variant\deploy_windows.ps1 `
    -Repo "https://github.com/USER/rag-chatbot.git" -AdminToken "пароль"
```

Клонирует репозиторий в `C:\rag-chatbot` и запускает установку.

## Управление

```powershell
powershell -File windows_variant\manage_windows.ps1 status   # статус задачи и контейнера
powershell -File windows_variant\manage_windows.ps1 start|stop|restart|logs
```

## Особенности и ограничения

- **Qdrant.** На Windows запускается в Docker Desktop. Для автозапуска после
  перезагрузки включите в Docker Desktop «Start when you log in», а контейнер
  поднят с `--restart unless-stopped`.
- **Автозапуск приложения.** Через Scheduled Task `ONSTART` от SYSTEM. Кнопка
  «Перезапустить сервис» в админке (она завершает процесс) на Windows не поднимет
  его автоматически — используйте `manage_windows.ps1 restart`.
- **Whisper.** Используется `faster-whisper` (кросс-платформенный); `mlx-whisper`
  (Apple) на Windows не нужен.
- **Удалённые хосты, дообучение, vLLM.** Дообучение и vLLM рассчитаны на Linux+GPU;
  на Windows доступны вектор-поиск, граф-RAG (LightRAG) и Ollama-генерация.
- **PowerShell не тестировался на реальном сервере** — при первом запуске возможны
  правки под конкретную версию Windows/winget; ошибки видны в выводе и в
  `%TEMP%\rag_api.log`.
