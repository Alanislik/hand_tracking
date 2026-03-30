# Hand Tracking: Gesture Photo Editor

Проект ориентирован на бесконтактную обработку фото жестами (OpenCV + Pillow), с запуском на Kali Linux, Windows и macOS.

## Основной сценарий
- Поворот ладони -> поворот изображения
- Разведение/сведение большого и указательного пальцев -> масштаб
- Горизонтальный свайп ладонью -> переключение фильтров (`BW`, `SEPIA`, `BLUR`)

## Быстрый старт

### 1) Подготовка окружения
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Запуск фоторедактора (основной режим)
```bash
python gesture2/main.py --app editor --image /path/to/photo.jpg
```

Если `--image` не указан, откроется встроенный demo-canvas.

При первом запуске автоматически скачаются модели MediaPipe в `gesture2/models/`.

## Клавиши фоторедактора
- `Q` / `Esc` — выход
- `R` — сброс поворота и масштаба
- `S` — сохранить текущий результат в PNG
- `[` / `]` — ручное переключение фильтров

## Точность распознавания
- Поворот работает в позе открытой ладони.
- Масштаб работает в позе `thumb + index` (остальные пальцы опущены), чтобы снизить ложные срабатывания.
- Свайп имеет cooldown и срабатывает только при открытой ладони.

## Legacy режим (курсор)
Старый режим управления курсором сохранён:
```bash
python gesture2/main.py --app cursor
```

## Кроссплатформенные заметки
- Kali/Linux: обычно работает из коробки, нужен доступ к камере.
- Windows: при необходимости запускайте терминал с обычными правами; для системных действий курсора могут требоваться повышенные права в legacy-режиме.
- macOS: разрешите доступ к камере для терминала/IDE.

## Установка на разных ОС (важно)
Возможны отличия из-за wheel-пакетов `mediapipe`/`opencv-python` и версии Python.

Рекомендуемый Python: `3.10` или `3.11`.

- Kali/Linux:
```bash
sudo apt update
sudo apt install -y python3-venv python3-dev libgl1 libglib2.0-0
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

- Windows (PowerShell):
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

- macOS:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Если установка `mediapipe` не проходит:
1. Проверьте версию Python (`python --version`).
2. Пересоздайте окружение на `3.11`.
3. Обновите pip/setuptools/wheel и повторите установку.

## Совместная разработка (GitHub Flow)
1. Создайте ветку: `git checkout -b feature/<короткое-имя>`
2. Делайте маленькие коммиты с понятными сообщениями
3. Откройте Pull Request в `main`
4. Перед PR запустите локальную проверку:
```bash
python -m py_compile $(find gesture2 -name '*.py')
```

Подробности — в `CONTRIBUTING.md`.
