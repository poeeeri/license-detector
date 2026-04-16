# License Plate Detection

  

**Авторы:**

- [Доржиева Пурбо-Суруна](https://github.com/poeeeri) (группа 972401)

- [Норец Елена](https://github.com/monalenka) (группа 972402)

  

## Установка и обучение модели

После клонирования репозитория установите зависимости через Poetry:

```
poetry install
```

Перед обучением нужно загрузить датасет.

на Linux:

```
cd data
curl -L "https://app.roboflow.com/ds/lVYKZyO4yr?key=aEs0sk1PwW" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
```

на Windows
```
cd data
Invoke-WebRequest "https://app.roboflow.com/ds/lVYKZyO4yr?key=aEs0sk1PwW" -OutFile roboflow.zip
Expand-Archive roboflow.zip -DestinationPath .
Remove-Item roboflow.zip
```


### Обучение на локальном хосте

Запустите ClearML-сервер, выполнив

```
docker-compose -f clearml-server-new/docker-compose.yml up -d
```

После запуска будет доступно
- Web UI: http://localhost:18080
- API: http://localhost:18008
- File server: http://localhost:18081

Для запуска обучения нужно использовать команду:
```
poetry run python scripts/train.py --data data/data.yaml --device cpu
```

В целом, у запуска обучения может быть ряд параметров:

| Параметр            | Описание                        | По умолчанию               |
|---------------------|---------------------------------|----------------------------|
| `--data`            | путь к датасету                 | **надо указать**           |
| `--model`           | название модели для дообучения  | `yolov8n.pt`               |
| `--epochs`          | количество эпох                 | `30`                       |
| `--imgsz`           | размер изображения              | `640`                      |
| `--batch`           | размер одного батча             | `16`                       |
| `--device`          | Устройство: `cpu` или `cuda`    | `cpu`                      |
| `--project`         | название проекта в ClearML      | `license-detector`         |
| `--task-name`       | название задачи в ClearML       | `train-license-plate-detector` |
| `--experiment-name` | название эксперимента в ClearML | `yolo-finetune`            |


## Запуск evaluation
Для оценки качества модели на validation/test split используется стандартная валидация Ultralytics YOLO.
```powershell
poetry run yolo detect val model=weights/best.pt data=data/data.yaml split=test device=cpu
```

для гпу 
```powershell
poetry run yolo detect val model=weights/best.pt data=data/data.yaml split=test device=cuda
```


## Запуск детекции

После обучения веса должны сохраниться в папку /weights.
Для обработки видео ввести команду

```
poetry run python scripts/run.py --mode video --input videos/road.mp4 --output result.mp4 --weights weights/best.pt
```

![Детекция на видео](demo/road_video.gif)

Для потокового режима

```
poetry run python scripts/run.py --mode cam --cam_id 0 --weights weights/best.pt
```

![Детекция на камере](demo/demo_cam.gif)
  
### Параметры запуска
В целом, у запуска может быть ряд параметров:

| Параметр    | Описание                       | По умолчанию      |
|-------------|--------------------------------|-------------------|
| `--mode`    | Режим: `video` или `cam`       | **надо указать**  |
| `--input`   | Путь к видеофайлу (для video)  | **надо указать**  |
| `--output`  | Путь для сохранения результата | `output.mp4`      |
| `--weights` | Путь к весам модели            | `weights/best.pt` |
| `--device`  | Устройство: `cpu` или `cuda`   | автоопределение   |
| `--conf`    | Порог уверенности              | `0.25`            |
| `--cam_id`  | ID веб-камеры (для cam)        | `0`               |

### Запуск через Docker

Кроме того возможно запустить детекцию номерных знаков в Docker-контейнере без установки зависимостей на хосте.

Сборка образа и запуск обработки видео videos/road.mp4 (по умолчанию):
```
docker-compose build
docker-compose up
```
Обработка другого файла
```
docker-compose run --rm license-detector \
  --mode video --input videos/new_video.mp4 --output output/my_result.mp4
```

Запуск режима веб-камеры. Обратите внимание, что из-за ограничений Docker Desktop на Windows режим камеры не работает
```
docker-compose run --rm --device /dev/video0 license-detector \
  --mode cam --cam_id 0 --weights weights/best.pt
```

Параметры запуска через Docker работают так же, как при локальном запуске - смотрите [Параметры запуска](#параметры-запуска).


## Использованные ресурсы

- Ultralytics YOLOv8
- ClearML