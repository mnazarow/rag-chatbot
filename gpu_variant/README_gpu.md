# GPU-вариант (Linux + NVIDIA)

Тот же RAG, но генерация идёт через **vLLM** (CUDA) вместо Ollama, а эмбеддинги,
реранк и транскрибация работают на GPU. Код приложения общий с основным проектом —
меняются только бэкенд генерации и инфраструктура (через `.env`).

## Чем отличается от Apple-варианта

| Компонент      | Apple (Mac Studio)        | GPU-сервер (этот вариант)            |
|----------------|---------------------------|--------------------------------------|
| Генерация LLM  | Ollama (Metal)            | **vLLM**, OpenAI-совместимый API     |
| `LLM_BACKEND`  | `ollama`                  | `openai`                             |
| Эмбеддинги/реранк | MPS                    | **CUDA**                             |
| Транскрибация  | mlx-whisper               | **faster-whisper** (CUDA)            |
| ОС             | macOS arm64               | Ubuntu 22.04/24.04                   |

Почему vLLM: высокий throughput и батчинг под одновременные запросы многих
сотрудников, continuous batching, paged attention. Для команды это заметно
производительнее, чем Ollama.

## Подбор модели под GPU

vLLM грузит модель целиком в видеопамять. Ориентир по VRAM (квантизация AWQ/GPTQ):

| GPU (VRAM)          | Рекомендуемая модель                  | Настройки                         |
|---------------------|---------------------------------------|-----------------------------------|
| 24 ГБ (RTX 4090/3090)| `Qwen/Qwen2.5-14B-Instruct-AWQ`      | TP=1, max-len 16k (по умолчанию)  |
| 48 ГБ (A6000/L40S)  | `Qwen/Qwen2.5-32B-Instruct-AWQ`       | TP=1                              |
| 80 ГБ (A100/H100)   | `Qwen/Qwen2.5-72B-Instruct-AWQ`       | TP=1                              |
| 2×24–48 ГБ          | 32B/72B без квантизации               | `VLLM_TP=2` (tensor-parallel)     |

Меняется через env перед запуском, напр.:
```bash
VLLM_MODEL=Qwen/Qwen2.5-32B-Instruct-AWQ VLLM_TP=1 sudo bash setup_gpu.sh
```
И тот же `LLM_MODEL` в `.env` должен совпадать с `VLLM_MODEL`.

## Установка — одной командой (рекомендуется)

На чистом сервере с драйвером NVIDIA (проверьте `nvidia-smi`):

```bash
cd gpu_variant
ADMIN_TOKEN='придумайте-пароль' sudo -E bash run_gpu.sh
```

`run_gpu.sh` делает всё сам: Docker + NVIDIA Container Toolkit, поднимает
vLLM + Qdrant, ставит Python-окружение с CUDA-torch, регистрирует
**systemd-сервис `rag-api`** с автозапуском и `Restart=always`. После этого
**все остальные настройки делаются в веб-панели** — править `.env` вручную не нужно.

Можно переопределить стартовую модель:
```bash
VLLM_MODEL=Qwen/Qwen2.5-32B-Instruct-AWQ VLLM_TP=1 ADMIN_TOKEN='...' sudo -E bash run_gpu.sh
```

По завершении откройте `http://<ip-сервера>:8000` → раздел **«Администратор»**.

### Что настраивается в админ-панели

Все параметры системы редактируются в веб-интерфейсе и применяются по-разному:

- **на лету** — параметры поиска (TOP_K, порог), температура, имя модели для
  запросов, системный промпт, авто-фильтр;
- **после переиндексации** (кнопка «Переиндексировать») — папка с документами
  `DOCS_DIR`, размер/перекрытие чанков, бэкенд и модель Whisper;
- **рестарт vLLM** (кнопка «Применить модель LLM») — модель контейнера
  `VLLM_MODEL`, длина контекста, число GPU (`VLLM_TP`);
- **после перезапуска сервиса** (кнопка «Перезапустить сервис») — модели
  эмбеддингов/реранка, устройство, адрес и коллекция Qdrant, бэкенд LLM.

Панель «Состояние и операции» показывает доступность Qdrant и LLM, число чанков
в базе и статус индексации.

### Управление

```bash
bash gpu_variant/manage.sh status      # статус сервиса и контейнеров
bash gpu_variant/manage.sh logs        # логи API
bash gpu_variant/manage.sh vllm-logs   # логи vLLM
bash gpu_variant/manage.sh restart|stop|start
```

### Ручная установка (без systemd)

Остаётся доступной через `setup_gpu.sh` + ручной `uvicorn` — см. историю файла;
для продакшена предпочтительнее `run_gpu.sh`.

## Приватные/гейтед модели

Если модель требует токен Hugging Face, экспортируйте `HF_TOKEN` перед запуском —
он пробрасывается в контейнер vLLM (см. `docker-compose.gpu.yml`).

## Проверка

```bash
curl http://localhost:8001/v1/models                 # vLLM поднялся, модель видна
curl http://localhost:6333/healthz                   # Qdrant жив
curl -X POST http://localhost:8000/chat -H 'Content-Type: application/json' \
     -d '{"question":"тестовый вопрос"}'             # сквозной ответ RAG
```

## Заметки

- LightRAG-ветка тоже работает на GPU: укажите в её `.env` `LLM_BACKEND` не нужен
  (LightRAG ходит в Ollama напрямую) — для чистого vLLM-сервера используйте
  OpenAI-функции LightRAG (`openai_complete`) вместо ollama; при необходимости
  поправьте `lightrag_variant/rag_lightrag.py`.
- Версии образов (`vllm-openai:v0.6.6`, `qdrant:v1.12.4`) зафиксированы; при
  обновлении сверяйте флаги командной строки vLLM — они меняются между релизами.
