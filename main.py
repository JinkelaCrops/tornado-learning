# from utils.datatool import RemoteIO
#
# remoteio = RemoteIO("localhost")
#
# remoteio.test(1)

# import os
# os.chdir("PycharmProjects/tornado-learning")
# from filedown.app.biz import sleeeep

# @gen.coroutine
# def sleeeep(args):
#     print("i'm sleeping")
#     time.sleep(10)
#     print("wake up")
#     for next in args:
#         if next != "m":
#             yield "in the sleep %s" % next
#         else:
#             print(1)


from tornado import gen
from tornado.ioloop import IOLoop
from tornado.queues import Queue

q = Queue(maxsize=2)


@gen.coroutine
def consumer():
    count = 0
    while True:
        if q.empty() and count == 1:
            print("stop")
            count -= 1
        else:
            count = q.qsize()
            item = yield q.get()
            print('Doing work on %s' % item)
            yield gen.sleep(0.5)


@gen.coroutine
def producer():
    for item in range(5):
        yield q.put(item)
        print('Put %s' % item)
    for item in range(5):
        yield q.put(item)
        print('Put %s' % item)


for item in range(5):
    q.put(item)


@gen.coroutine
def main():
    # Start consumer without waiting (since it never finishes).
    IOLoop.current().spawn_callback(consumer)
    # yield producer()  # Wait for producer to put all tasks.
    print("abc")
    yield q.join()  # Wait for consumer to finish all tasks.
    print('Done')


IOLoop.current().run_sync(main)
