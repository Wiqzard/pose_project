from engine.validator import BaseValidator


class MposeValidator(BaseValidator):
    def __init__(
        self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None
    ):
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)

        self.metrics = None
        
        
    def get_desc(self):
        return ("%22s" + "%11s" * 6) % (
            "Class",
            "Images",
            "Instances",
            "Pose(TLoss",
            "Ang",
            "Trans",
            "Fitness)",
        )

    def init_metrics(self, model):
        """Initialize evaluation metrics for YOLO."""
        val = self.data.get(self.args.split, '')  # validation path
        self.args.save_json =  not self.training  # run on final val if training 
        self.names = model.names
        self.nc = len(model.names)
        self.metrics.names = self.names
        #self.metrics.plot = self.args.plots
        #self.confusion_matrix = ConfusionMatrix(nc=self.nc)
        self.seen = 0
        self.jdict = []
        self.stats = []