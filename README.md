# Neural Racing Lab

Учебно-исследовательский проект: автономное вождение в CarRacing-v3.
Текущий этап: **PHASE 0** — воспроизводимая среда и случайный агент.
Реализация PPO и CNN появится на следующих этапах.

Проверено 2026-10-27: 16 тестов, Ruff, запуск CarRacing и реальная операция CUDA.
Первое локальное демо: `artifacts/phase0-random/episode-0.mp4` (20.02 с).
Результаты и ограничения проверки — в [журнале экспериментов](docs/EXPERIMENTS.md).

## Запуск (Windows / PowerShell)

Требуются uv и Python 3.12.14. Все пакеты устанавливаются в `.venv`.

```powershell
git clone https://github.com/miradimo/neural-racing-lab.git
cd neural-racing-lab
uv sync --locked --extra torch
uv run --locked --extra torch nrl-check --output artifacts/diagnostics.json
uv run --locked --extra torch nrl-random --episodes 3 --seed 42 --video
uv run --locked --extra torch nrl-manual --seed 42
```

Extra `torch` устанавливает CUDA 12.6 сборку PyTorch из официального индекса.
Без GPU-проверки достаточно `uv sync --locked` и команд без `--extra torch`.
Если extra установлен, продолжайте указывать его в `uv run`/`uv sync`, чтобы
uv сохранял выбранный набор зависимостей. Активация окружения не обязательна.
В VS Code выберите `.venv/Scripts/python.exe`.

Для random-agent можно включить `--human` вместо `--video`. Видео записывается
для первого эпизода; результаты всех эпизодов — в `metrics.json`.
`--output artifacts/my-run` задаёт новый каталог; существующий не перезаписывается.
`--max-episode-steps 1000` задаёт TimeLimit; более короткие запуски пригодны для
smoke tests, но их reward нельзя напрямую сравнивать со стандартным лимитом.

Ручное управление: стрелки/WASD, пробел — тормоз, R — заново та же трасса,
Esc — выход. Здесь ровно те же пять дискретных действий, что у random-agent:
0 — ничего, 1 — вправо, 2 — влево, 3 — газ, 4 — тормоз.
Одновременно газ и поворот невозможны: тормоз имеет приоритет, затем поворот,
затем газ. Это ограничение выбранного action space, не ошибка клавиатуры.

## Проверки

```powershell
uv run --locked --extra torch ruff check .
uv run --locked --extra torch ruff format --check .
uv run --locked --extra torch pytest
```

Тесты выполняют короткие реальные запуски Box2D, проверяют seeds, TimeLimit,
флаги завершения, запись и декодирование видео. Они не запускают обучение.

## Где читать код

1. `envs.py`: единая конфигурация среды и два независимых RNG.
2. `random_agent.py`: reset → action → step → reward → проверка конца эпизода.
3. `manual_control.py`: та же среда с действиями от клавиатуры.
4. `diagnostics.py`: проверка реального окружения и вычислений CUDA.

Пакет расположен в `src/neural_racing_lab/`; команды CLI объявлены в
`pyproject.toml`. Наблюдения остаются RGB uint8, reward не изменяется.
Метрика episode return — сумма наград до terminated или truncated.
Случайный агент служит проверкой инфраструктуры, а не обученной моделью.

## Документация

- [Прогресс и следующие шаги](docs/PROGRESS.md)
- [Дорожная карта](docs/ROADMAP.md)
- [Теория первого занятия](docs/LEARNING_NOTES.md)
- [Решения](docs/DECISIONS.md)
- [Эксперименты](docs/EXPERIMENTS.md)

Фиксация зависимостей: `pyproject.toml` + `uv.lock` + `.python-version`.
Видео, логи, данные и модели исключены из Git. Commit/push — только по запросу.
