import threading
from random import randint

# semaphore -> cheio, vazio
# lock -> mutex


class Produtor(threading.Thread):
    def __init__(self, buffer, bufferSize, index, cheio, vazio, mutex):
        threading.Thread.__init__(self)
        self._buffer = buffer
        self._bufferSize = bufferSize
        self._index = index  # Index do item que esta sendo produzido
        self._cheio = cheio
        self._vazio = vazio
        self._mutex = mutex

    def run(self):
        # for _ in range(10):
        while(True):
            self._vazio.acquire()
            self._mutex.acquire()

            item = randint(0, 20)
            self._buffer[self._index] = item
            self._index = (1 + self._index) % self._bufferSize

            print(self._buffer)

            self._mutex.release()
            self._cheio.release()


class Consumidor(threading.Thread):
    def __init__(self, buffer, bufferSize, index, cheio, vazio, mutex):
        threading.Thread.__init__(self)
        self._buffer = buffer
        self._bufferSize = bufferSize
        self._index = index  # Index do item que esta sendo consumido
        self._cheio = cheio
        self._vazio = vazio
        self._mutex = mutex

    def run(self):
        # for _ in range(10):
        while(True):
            self._cheio.acquire()
            self._mutex.acquire()

            print("Elemento consumido: ", self._buffer[self._index])

            self._buffer[self._index] = None
            self._index = (1 + self._index) % self._bufferSize

            self._mutex.release()
            self._vazio.release()

def main():
    bufferSize = 8
    buffer = []
    # Inicializando o buffer
    for _ in range(bufferSize):
        buffer.append(None)

    vazio = threading.Semaphore(bufferSize)
    cheio = threading.Semaphore(0)
    mutex = threading.Lock()

    indexProdutor = 0
    indexConsumidor = 0

    t1 = Produtor(buffer=buffer, bufferSize=bufferSize, index=indexProdutor,
                  cheio=cheio, vazio=vazio, mutex=mutex)
    t2 = Consumidor(buffer=buffer, bufferSize=bufferSize, index=indexConsumidor,
                    cheio=cheio, vazio=vazio, mutex=mutex)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
if __name__ == '__main__':
    main()
