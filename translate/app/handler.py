# -*-coding: utf-8 -*-
# __author__ : tinytiger
# __time__   : '2018/1/4 11:08'

import json
from json import JSONDecodeError

# If this runs wrong, don't ask me, I don't know why;
# If this runs right, thank god, and I don't know why.
# Maybe the answer, my friend, is blowing in the wind.
import tornado.web
from tornado import gen

from app.biz import Consumer
from app.biz import decoder
from app.biz import Sync_io
from app.biz import understand_data
from app.loggerInst import log

sync_io = Sync_io()


class WriteDownHandler(tornado.web.RequestHandler):

    def decode_argument(self, value, name=None):
        from tornado.escape import _unicode
        from tornado.web import HTTPError
        try:
            return _unicode(value)
        except UnicodeDecodeError:
            return bytes(value)
        except:
            raise HTTPError(400, "Invalid unicode or bytes in %s: %r" %
                            (name or "url", value[:40]))

    def post(self):
        log.info("[handler] WriteDownHandler.post: get data from memory")
        data = self.get_argument("data")
        data = understand_data(data)
        log.info("[handler] WriteDownHandler.post: got data")

        file_name = self.get_argument("file_name")
        if not isinstance(file_name, str):
            log.warn("[handler] WriteDownHandler.post: TypeError: file_name is not string")
            return

        # path = data_dir % file_name
        path = file_name
        write_flag = self.get_argument("write_mode")
        if not isinstance(write_flag, str):
            log.warn("[handler] WriteDownHandler.post: TypeError: write_flag is not string")
            return

        log.info("[handler] WriteDownHandler.post: write data into %s" % path)

        # with open(path, write_flag, encoding="utf8") as f:
        #     f.writelines(data)

        sync_io.write(path, write_flag, data)
        log.info("[handler] WriteDownHandler.post: wrote done")


class SendOutHandler(tornado.web.RequestHandler):

    def post(self):
        file_name = self.get_argument("file_name")
        if not isinstance(file_name, str):
            log.warn("[handler] SendOutHandler.post: TypeError: file_name is not string")
            return

        # path = data_dir % file_name
        path = file_name
        read_start = int(self.get_argument("read_start"))
        read_nrows = int(self.get_argument("read_nrows"))
        pointer = int(self.get_argument("pointer"))

        if pointer:
            log.info("[handler] SendOutHandler.post: send data from %s [%s] -> %s" % (path, read_start, read_nrows))
        else:
            log.info("[handler] SendOutHandler.post: send data from %s [%s : %s]" % (
                path, read_start, read_start + read_nrows))

        # with open(path, "r", encoding="utf8") as f:
        #     tmp = json.dumps({"data": line_read(f, read_start, read_nrows)})
        # self.write(tmp)

        self.write(sync_io.read(path, read_start, read_nrows, pointer))
        log.info("[handler] SendOutHandler.post: sent data")


class ExecHandler(tornado.web.RequestHandler):

    def post(self):
        python_code = self.get_argument("python_code")
        result_name = self.get_argument("result_name")
        if not isinstance(python_code, str):
            log.warn("[handler] TypeError: python_code is not string")
            return
        if not isinstance(result_name, str):
            log.warn("[handler] TypeError: result_name is not string")
            return

        log.info("[handler] execute python code: %s" % python_code)
        # before_exec = list(vars().keys())
        # exec(python_code)
        # after_exec = list(vars().keys())
        # tmp_vars = list(filter(lambda x: not x.startswith("_"), set(after_exec) - set(before_exec)))
        # self.write(json.dumps({"result": vars()[tmp_vars[0]]}))
        exec(python_code)
        self.write(json.dumps({"result": vars()[result_name]}))
        log.info("[handler] executed")


class Test1(tornado.web.RequestHandler):

    def post(self):
        data = self.get_argument("args")
        data = int(data)
        log.info("[handler] Test1.post: get data and put into input queue")
        Consumer.q.put(data)


class Test2(tornado.web.RequestHandler):

    def post(self):
        log.info("[handler] Test2.post: send data from output queue")
        data = None if Consumer.p.empty() else Consumer.p.get()
        self.write(json.dumps({"result": data}))
        if Consumer.p.empty():
            log.info("[handler] Test2.post: empty output queue")
        else:
            log.info("[handler] Test2.post: " + data)


class TranslateSend(tornado.web.RequestHandler):

    def post(self):
        data = self.get_argument("args")
        log.info("[handler] TranslateSend.post: get sentence and put into input queue")
        single_decode.q.put(data)


class TranslateGet(tornado.web.RequestHandler):

    def post(self):
        log.info("[handler] TranslateSend.post: send translated sentence from output queue")
        data = None if single_decode.p.empty() else single_decode.p.get()
        self.write(json.dumps({"result": data}))
        if single_decode.p.empty():
            log.info("[handler] TranslateSend.post: empty output queue")
        else:
            log.info("[handler] TranslateSend.post: " + data)


class Translate(tornado.web.RequestHandler):
    @gen.coroutine
    def post(self):
        data = self.get_argument("args")
        datas = json.loads(data)
        if not isinstance(datas, list) or not all(map(lambda x: isinstance(x, str), datas)):
            log.warn("[handler] Translate.post: invalid input")
            self.write(json.dumps({"result": "invalid input"}))
        else:
            log.info("[handler] Translate.post: get sentence and put into input queue")
            # log.info("[handler] Translate.post: " + data)
            decoder.q.put(datas)
            try:
                result = yield decoder.p.get()
                self.write(json.dumps({"result": result}))
                log.info("[handler] Translate.post: send translated sentence")
            except Exception as e:
                log.warn("[handler] Translate.post: translate failed %s %s" % (e.__class__, e.__context__))
                self.write(json.dumps({"result": "translate failed"}))
