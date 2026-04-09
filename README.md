# License Plate Detection

**Авторы:** 
- [Доржиева Пурбо-Суруна](https://github.com/poeeeri) (группа 972401)
- [Норец Елена](https://github.com/monalenka) (группа 972402)

### Запуск с готовыми весами
После клонирования репозитория установить зависимости через Poetry:
```
poetry install
```
Скачать веса модели `best.pt`и поместить в папку `weights/`

Для обработки видео ввести команду
```
poetry run python run.py --mode video --input road.mp4 --output result.mp4 --weights weights/best.pt
```

Для потокового режима
```
poetry run python run.py --mode cam --cam_id 0 --weights weights/best.pt
```

В целом, у запуска может быть ряд параметров:

| Параметр    | Описание                       | По умолчанию      |
| ----------- | ------------------------------ | ----------------- |
| `--mode`    | Режим: `video` или `cam`       | надо указать      |
| `--input`   | Путь к видеофайлу (для video)  | надо указать      |
| `--output`  | Путь для сохранения результата | `output.mp4`      |
| `--weights` | Путь к весам модели            | `weights/best.pt` |
| `--device`  | Устройство: `cpu` или `cuda`   | автоопределение   |
| `--conf`    | Порог уверенности              | `0.25`            |
| `--cam_id`  | ID веб-камеры (для cam)        | `0`               |