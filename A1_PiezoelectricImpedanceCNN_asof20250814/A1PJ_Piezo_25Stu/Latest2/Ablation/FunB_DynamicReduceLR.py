# FunB_DynamicReduceLR.py
# dynamic_lr_scheduler.py
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K

class DynamicPatienceReduceLROnPlateau(Callback):
    """
    自定义学习率调度器：每次学习率下降后，patience 按 factor 增加，
    并模仿 ReduceLROnPlateau 的日志输出格式，同时显示当前 patience。
    """
    def __init__(self, monitor='val_loss', factor=0.8, initial_patience=32,
                 min_lr=5e-4, verbose=0):
        super(DynamicPatienceReduceLROnPlateau, self).__init__()
        self.monitor = monitor
        self.factor = factor
        self.patience = initial_patience
        self.initial_patience = initial_patience
        self.min_lr = min_lr
        self.verbose = verbose
        self.wait = 0
        self.best = float('inf')
        self.reductions = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return

        if current < self.best:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                old_lr = float(K.get_value(self.model.optimizer.lr))
                if old_lr > self.min_lr:
                    new_lr = max(old_lr * self.factor, self.min_lr)
                    K.set_value(self.model.optimizer.lr, new_lr)
                    self.wait = 0
                    self.reductions += 1
                    self.patience = max(1, int(self.patience / self.factor))  # 至少为1

                    if self.verbose > 0:
                        # 模仿原生 ReduceLROnPlateau，同时显示 patience
                        print(f"\nEpoch {epoch + 1:05d}: ReduceLROnPlateau reducing learning rate "
                              f"to {new_lr:.10e} (current patience: {self.patience}).")


