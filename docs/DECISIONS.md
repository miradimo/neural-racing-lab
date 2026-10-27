# Архитектурные решения

## 2026-10-27 — PHASE 0

1. **Формат обучения.** Ассистент пишет реализацию, объясняет теорию на коде и
   результатах. Пользователь задаёт вопросы по непонятным местам. Самостоятельное
   написание ключевых алгоритмов не является обязательным заданием.
2. **Python 3.12.14 + uv.** Этот Python уже установлен. Окружение локальное `.venv`,
   зависимости воспроизводятся через uv.lock. Глобальные окружения не меняются.
3. **Gymnasium 1.4.0 [box2d].** PyPI metadata подтверждает поддержку Python 3.12
   и Box2D 2.3.10 с Windows wheel. Не нужны глобальный SWIG и C++ build tools.
4. **PyTorch отдельным extra.** Для проверки CUDA используем 2.12.1 + cu126 из
   официального индекса; этот вариант документирован для Windows. Фиксируем
   конкретную поддерживаемую сборку, а не автоматически самый новый релиз.
   Random-agent работает и без torch. torchvision/torchaudio пока не нужны.
5. **Один env, discrete actions, raw RGB, исходный reward.** Не добавляем wrappers,
   frame skip, reward shaping, multiprocessing и обучение до следующих фаз.
6. **Конфигурация через CLI.** На этом этапе достаточно seed, episodes,
   max_episode_steps, render mode и output. YAML и конфигурационный framework
   не нужны. Параметры и версии сохраняются в metrics.json.
7. **Видеозапись imageio + FFmpeg.** Кадры передаются в encoder последовательно,
   без накопления целого ролика в RAM. Сохраняется первый эпизод.
8. **Manual control владеет окном.** Среда отдаёт rgb_array, pygame обрабатывает
   события и выводит кадр. Так два обработчика не конкурируют за очередь событий.
9. **Никаких научных выводов по smoke baseline.** Seeds 42–44 используются для
   проверки инфраструктуры. Они не объявляются независимым test set будущей модели.

Источники для выбора зависимостей:
- https://pypi.org/pypi/gymnasium/json
- https://pypi.org/pypi/Box2D/json
- https://pytorch.org/get-started/previous-versions/
- https://gymnasium.farama.org/environments/box2d/car_racing/
