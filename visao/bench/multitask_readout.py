import numpy as np

class MultiTaskReadout:
    """Readout com pesos separados por tarefa.
    
    Cada tarefa tem seu próprio W_out e b_out, eliminando interferência
    entre tarefas no Permuted-MNIST (20 tarefas, 10 classes cada).
    """
    def __init__(self, n_hidden, n_out, lr=0.1, seed=0):
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.lr = lr
        self.rng = np.random.default_rng(seed)
        self.readouts = {}
    
    def get(self, task_id):
        if task_id not in self.readouts:
            scale = 1.0 / np.sqrt(self.n_hidden)
            W = self.rng.normal(0, scale, (self.n_out, self.n_hidden))
            b = np.zeros(self.n_out)
            self.readouts[task_id] = (W, b)
        return self.readouts[task_id]
    
    def predict(self, x, task_id):
        W, b = self.get(task_id)
        return W @ x + b
    
    def update(self, x, target, task_id, lr_factor=1.0):
        W, b = self.get(task_id)
        pred = W @ x + b
        err = target - pred
        W += self.lr * lr_factor * np.outer(err, x)
        b += self.lr * lr_factor * err
        self.readouts[task_id] = (W, b)
        return float((err ** 2).mean())