from collections import deque
import numpy as np


class EmgVisualizer:
    def __init__(self,channels=4,history=1000):
        import matplotlib.pyplot as plt
        self.plt=plt; plt.ion(); self.fig,(self.ax_raw,self.ax_processed,self.ax_pred)=plt.subplots(3,1); self.raw=[deque(maxlen=history) for _ in range(channels)]; self.processed=[deque(maxlen=history) for _ in range(channels)]
    def update(self,raw,processed,prediction=None,confidence=None):
        for series,value in zip(self.raw,raw): series.append(value)
        for series,value in zip(self.processed,processed): series.append(value)
        self.ax_raw.clear()
        for series in self.raw: self.ax_raw.plot(series)
        self.ax_processed.clear()
        for series in self.processed: self.ax_processed.plot(series)
        self.ax_pred.clear()
        if prediction is not None: self.ax_pred.bar(np.arange(7),prediction); self.ax_pred.set_ylim(0,1); self.ax_pred.set_title(f"prediction / confidence {confidence if confidence is not None else 'n/a'}")
        self.fig.canvas.draw_idle(); self.plt.pause(.001)
