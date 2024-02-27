import time

class Temporizador:
    def __init__(self):
        self.inicio = None
    
    def iniciar(self):
        self.inicio = time.time()
    
    def tiempo_transcurrido(self):
        if self.inicio is None:
            return 0
        return round(time.time() - self.inicio, 2)