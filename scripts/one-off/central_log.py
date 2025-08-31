import logging
import logging.handlers
import multiprocessing
import os

class CentralLoggerMain:
    def __init__(self, log_path="main.log", num_workers=2):
        self.log_queue = multiprocessing.Queue()
        self.logger = logging.getLogger("main")
        self.logger.setLevel(logging.INFO)
        self.file_handler = logging.FileHandler(log_path)
        formatter = logging.Formatter('%(asctime)s %(processName)s %(levelname)s: %(message)s')
        self.file_handler.setFormatter(formatter)
        self.queue_listener = logging.handlers.QueueListener(self.log_queue, self.file_handler)
        self.num_workers = num_workers

    @staticmethod
    def worker(msg):
        logger = logging.getLogger("worker")
        logger.info(f"Subprocess log from PID {os.getpid()}: {msg}")

    def run(self):
        self.queue_listener.start()
        self.logger.addHandler(self.file_handler)
        self.logger.info("Main process started logging.")

        pool = multiprocessing.Pool(
            processes=self.num_workers,
            initializer=worker_initializer,
            initargs=(self.log_queue,)
        )
        for i in range(4):
            pool.apply_async(CentralLoggerMain.worker, args=(f"message {i}",))
        pool.close()
        pool.join()

        self.logger.info("Main process finished logging.")
        self.queue_listener.stop()

def worker_initializer(log_queue):
    logger = logging.getLogger("worker")
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    logger.setLevel(logging.INFO)
    handler = logging.handlers.QueueHandler(log_queue)
    logger.addHandler(handler)

if __name__ == "__main__":
    CentralLoggerMain().run()
