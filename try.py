from utils.datatool import RemoteIO

remoteio = RemoteIO("localhost")
remoteio.test("3")

from tornado.queues import Queue

q = Queue(maxsize=3)


def putq(q):
    q.put(3)


putq(q)
