import logging
import logging.handlers
import multiprocessing
import os

def worker(msg):
    logger = logging.getLogger("worker")
    logger.info(f"Subprocess log from PID {os.getpid()}: {msg}")

def worker_initializer(log_queue):
    logger = logging.getLogger("worker")
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    logger.setLevel(logging.INFO)
    handler = logging.handlers.QueueHandler(log_queue)
    logger.addHandler(handler)

def main():
    log_queue = multiprocessing.Queue()
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler("main.log")
    formatter = logging.Formatter('%(asctime)s %(processName)s %(levelname)s: %(message)s')
    file_handler.setFormatter(formatter)
    queue_listener = logging.handlers.QueueListener(log_queue, file_handler)
    queue_listener.start()

    logger.addHandler(file_handler)
    logger.info("Main process started logging.")

    # Use Pool with initializer
    pool = multiprocessing.Pool(
        processes=2,
        initializer=worker_initializer,
        initargs=(log_queue,)
    )
    for i in range(4):
        pool.apply_async(worker, args=(f"message {i}",))
    pool.close()
    pool.join()

    logger.info("Main process finished logging.")
    queue_listener.stop()

if __name__ == "__main__":
    main()
