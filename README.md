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
poetry run python scripts/run.py --mode video --input videos/road.mp4 --output result.mp4 --weights weights/best.pt
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

### Обучение на локальном хосте
перед обучением, нужно загрузить датасет.

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

для запуска обучения нужно использовать команду:
```
poetry run python scripts/train.py --data data/data.yaml --device cpu
```

В целом, у запуска может быть ряд параметров:

| Параметр    | Описание                       | По умолчанию      |
| ----------- | ------------------------------ | ----------------- |
| `--data`    | путь к датасету | надо указать      |
| `--model`   | название модели для дообучения  | `yolov8n.pt`      |
| `--epochs`  | количество эпох | 30      |
| `--imgsz` | размер изображения | 640 |
| `--batch` | размер одного батча | 16 |
| `--device`  | Устройство: `cpu` или `cuda`   | `cpu`   |
| `--project`    | название проекта в ClearML | license-detector |
| `--task-name`    | название задачи в ClearML | train-license-plate-detector |
| `--experiment-name`  | название эксперимента в ClearML | yolo-finetune |