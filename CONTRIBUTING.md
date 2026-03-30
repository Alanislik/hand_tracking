# Contributing Guide

## Ветки
- `main` — стабильная ветка
- рабочие ветки: `feature/*`, `fix/*`, `chore/*`

## Стиль коммитов
Рекомендуемый формат:
- `feat: ...`
- `fix: ...`
- `chore: ...`
- `docs: ...`
- `refactor: ...`

Примеры:
- `feat: add left-handed toggle hint to HUD`
- `fix: prevent accidental drag lock`

## Pull Request checklist
- Код запускается локально
- Нет лишних файлов в коммите
- Обновлена документация (если менялось поведение)
- Пройдена проверка:
```bash
python -m py_compile $(find gesture2 -name '*.py')
```

## Code review
- Минимум 1 reviewer
- Не мержить PR с красным CI
- Для спорных архитектурных решений добавлять краткое rationale в описание PR
