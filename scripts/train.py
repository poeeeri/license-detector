import argparse
from pathlib import Path
from clearml import Task
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--project", default="license-detector")
    parser.add_argument("--task-name", default="train-license-plate-detector")
    parser.add_argument("--experiment-name", default="yolo-finetune")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    task = Task.init(
        project_name=args.project,
        task_name=args.task_name,
        task_type=Task.TaskTypes.training,
    )
    task.connect(vars(args))

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="runs",
        name=args.experiment_name,
    )

    metrics = getattr(results, "results_dict", {}) or {}
    logger = task.get_logger()
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, (int, float)):
            logger.report_scalar(
                title="train_metrics",
                series=metric_name,
                value=float(metric_value),
                iteration=args.epochs,
            )

    best_model = Path("runs") / args.experiment_name / "weights" / "best.pt"
    if best_model.exists():
        task.upload_artifact("best_model", artifact_object=best_model)

    task.close()


if __name__ == "__main__":
    main()