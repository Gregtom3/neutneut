import tensorflow as tf
from ModelEcalPlotter import ModelEcalPlotter
from Evaluator import Evaluator

class PlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, X, y, misc, model, param_intersections, tB, tD, outdir=None):
        super().__init__()
        self.X = X
        self.y = y
        self.misc = misc
        self.model = model
        self.param_intersections = param_intersections
        self.tB = tB
        self.tD = tD
        self.outdir = outdir

    def on_epoch_end(self, epoch, logs=None):
        
        evaluator = Evaluator(self.X, 
                              self.y, 
                              self.misc, 
                              is_intersections=self.param_intersections)
        evaluator.load_model(self.model)
        evaluator.predict()
        evaluator.cluster(self.tB, self.tD)
        
        for n in range(self.X.shape[0]):
            plotter = ModelEcalPlotter(evaluator.get_event_dataframe(n))
            outfile = f"{self.outdir}/ECAL_{epoch:05}_ev{n}.png" 

            suptitle = f"Epoch {epoch:05}\nLoss = {logs.get('loss'):.4f}"
            plotter.plot_all(tD=evaluator.tD,out=outfile,suptitle=suptitle)

        print(f"End of epoch {epoch+1}")