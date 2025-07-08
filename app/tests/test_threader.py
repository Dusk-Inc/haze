import pytest
from app.src.threader.core import Threader
from time import sleep
from queue import Empty
from threading import Thread

@pytest.fixture
def threader() -> Threader:
    return Threader()

def test_enqueue_dequeue(threader: Threader):
    threader.enqueue(1)
    threader.enqueue(2)
    assert threader.dequeue() == 1
    assert threader.dequeue() == 2

def test_dequeue_empty_queue(threader: Threader):
    with pytest.raises(Empty):
        threader.dequeue()

def test_synchronized(threader: Threader):
    results = []

    @threader.synchronized
    def synchronized_function(x):
        sleep(0.01)
        results.append(x)

    threads = [Thread(target=synchronized_function, args=(i,)) for i in range(10)]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(results) == list(range(10))


def test_thread_safe_enqueue_dequeue(threader: Threader):
    result = []

    def enqueue_items():
        for i in range(5):
            threader.enqueue(i)

    def dequeue_items():
        for _ in range(5):
            result.append(threader.dequeue())

    enqueue_thread = Thread(target=enqueue_items)
    dequeue_thread = Thread(target=dequeue_items)

    enqueue_thread.start()
    dequeue_thread.start()

    enqueue_thread.join()
    dequeue_thread.join()

    # Check that all items were dequeued correctly
    assert sorted(result) == list(range(5))