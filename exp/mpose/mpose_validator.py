from engine.validator import BaseValidator


class MposeValidator(BaseValidator):
    def __init__(
        self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None
    ):
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)

    def get_desc(self):
        """Return a formatted string summarizing class metrics of YOLO model."""
        return ("%22s" + "%11s" * 6) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
        )
