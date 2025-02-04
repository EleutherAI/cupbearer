from pathlib import Path

from cupbearer.detectors import AnomalyDetector
from cupbearer.tasks import Task


def main(
    task: Task,
    detector: AnomalyDetector,
    save_path: Path | str | None,
    pbar: bool = False,
    batch_size: int = 1024,
    layerwise: bool = False,
    train_from_test: bool = False,
    answer_accuracy: bool = False,
):
    if answer_accuracy:
        task.test_data.return_labels = ['answer', 'anomaly']
    detector.set_model(task.model)

    if train_from_test:
        detector.eval(
            dataset=task.train_test_mix_data,
            pbar=pbar,
            save_path=Path(save_path) / "train_from_test",
            batch_size=batch_size,
            layerwise=layerwise,
        )

    return detector.eval(
        dataset=task.test_data,
        pbar=pbar,
        save_path=save_path,
        batch_size=batch_size,
        layerwise=layerwise,
    )
