# -*-coding: utf-8 -*-
# __author__ : tinytiger
# __time__   : '2018/1/4 11:07'

# If this runs wrong, don't ask me, I don't know why;
# If this runs right, thank god, and I don't know why.
# Maybe the answer, my friend, is blowing in the wind.

from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.web import Application
from tornado.options import define, options

from app import handler
from app import biz

define("port", default=8000, help="run on the given port", type=int)

Handlers = [(r"/wt", handler.WriteDownHandler),
            (r"/sd", handler.SendOutHandler),
            (r"/ex", handler.ExecHandler),
            (r"/test1", handler.Test1),
            (r"/test2", handler.Test2),
            ]

application = Application(
    Handlers
)

if __name__ == '__main__':
    server = HTTPServer(application)
    server.listen(options.port)

    # ioloop = IOLoop.current()
    IOLoop.current().spawn_callback(biz.Consumer.consume)
    IOLoop.current().start()
