# Журнал экспериментов

Обучение моделей ещё не запускалось. Не путать техническую проверку со сравнением
RL-алгоритмов. Здесь фиксируются только реально полученные результаты.

## Шаблон будущего эксперимента

- Гипотеза и изменяемый фактор.
- Версия кода, lockfile, оборудование.
- Бюджет environment steps, время, число parallel envs.
- Training seeds; validation seeds для настройки; held-out test seeds для итоговой оценки.
- Policy evaluation: deterministic/stochastic; число эпизодов и TimeLimit.
- Mean, sample std и ограничения статистики.
- Ссылки на конфигурации, checkpoints, графики и raw metrics.
- Вывод, альтернативные объяснения и следующий эксперимент.

## 2026-10-27 — PHASE 0 smoke baseline

- Policy: uniform random, Discrete(5).
- Seeds: 42, 43, 44; TimeLimit: 1000.
- Reward и наблюдения не изменены; domain_randomize=False.
- Цель: проверить reset/step, флаги конца, метрики и сохранение видео.

Команда:

```powershell
uv run --locked --extra torch nrl-random --episodes 3 --seed 42 --video --output artifacts/phase0-random
```

| Seed | Episode return | Шаги | terminated | truncated |
|---|---:|---:|---|---|
| 42 | -50.53003533568978 | 1000 | false | true |
| 43 | -57.09570957095796 | 1000 | false | true |
| 44 | -65.61604584527294 | 1000 | false | true |

Среднее: **-57.74726358397356**. Выборочное std (ddof=1): **7.564080925513746**.
Все эпизоды завершились по TimeLimit. Обучения не было.
Elapsed: 20.495 с на этой машине, включая reset и запись первого эпизода;
это не измерение чистой скорости симуляции и не прогноз времени обучения PPO.

Окружение: Python 3.12.14, Gymnasium 1.4.0, Box2D 2.3.10, NumPy 2.5.4,
pygame-ce 2.5.8; lockfile содержит все точные версии. Оборудование — PROGRESS.md.

Артефакты (локальные, исключены из Git):

- `artifacts/phase0-random/metrics.json` — параметры, версии и результаты эпизодов.
- `artifacts/phase0-random/episode-0.mp4` — 1001 кадр, 50 FPS, 600×400, 20.02 с.
- `artifacts/phase0-random/video-validation.json` — проверено число кадров,
  успешно декодированы начало/середина/конец.
- `artifacts/phase0-random/preview.png` — визуально проверенный кадр из середины видео.
- `artifacts/diagnostics.json` — среда и успешное матричное умножение CUDA.

Вывод: инфраструктура выполняет случайные эпизоды, сохраняет результаты и видео.
Эти три эпизода не дают оснований судить о качестве PPO или общей сложности обучения.
