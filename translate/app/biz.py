import json
import queue
import pickle
import re
import time
from concurrent.futures import ThreadPoolExecutor

from tornado import gen
from tornado.queues import Queue

from app.loggerInst import log

from app.my_t2t_decoder_active import SessFieldPredict
from app.my_t2t_decoder_active import text_encoder
from app.my_t2t_decoder_active import tf

import config


class Decode(object):
    def __init__(self, sess_field):
        self.q = Queue(maxsize=1000)
        self.p = Queue(maxsize=1000)
        self.sess_field = sess_field

    @staticmethod
    def batch_pad(nd):
        max_length = max(map(len, nd))
        pad_nd = [i + [text_encoder.PAD_ID] * (max_length - len(i)) for i in nd]
        return pad_nd

    @gen.coroutine
    def decode(self):
        log.info("[biz] Decode: model loading ... ")
        saver = tf.train.Saver()

        with tf.Session(config=self.sess_field.sess_config) as sess:
            # Load weights from checkpoint.
            log.info("[biz] Decode: restoring parameters")
            saver.restore(sess, self.sess_field.ckpt)
            log.info("[biz] Decode: model already loaded")
            while True:
                inputs = yield self.q.get()
                log.info("[biz] Decode: " + str(inputs))
                st_time = time.time()
                inputs_numpy = [self.sess_field.encoders["inputs"].encode(i) + [text_encoder.EOS_ID] for i in inputs]
                num_decode_batches = (len(inputs_numpy) - 1) // self.sess_field.batch_size + 1
                results = []
                for i in range(num_decode_batches):
                    input_numpy = inputs_numpy[i * self.sess_field.batch_size:(i + 1) * self.sess_field.batch_size]
                    inputs_numpy_batch = input_numpy + [[text_encoder.EOS_ID]] * (
                            self.sess_field.batch_size - len(input_numpy))
                    inputs_numpy_batch = self.batch_pad(inputs_numpy_batch)  # pad using 0
                    # log.info("[biz] Decode: " + str(inputs_numpy_batch))
                    feed = {self.sess_field.inputs_ph: inputs_numpy_batch}
                    result = sess.run(self.sess_field.prediction, feed)
                    decoded_outputs = [self.sess_field.encoders["targets"].decode(i).strip("<pad>").strip("<EOS>") for i
                                       in result["outputs"][:len(input_numpy)]]
                    results += decoded_outputs
                self.p.put(results)
                log.info("[biz] Decode: source: " + str(inputs))
                log.info("[biz] Decode: target: " + str(results))
                log.info("[biz] Decode: using %s s" % (time.time() - st_time))


sess_field = SessFieldPredict(config.BATCH_SIZE)
decoder = Decode(sess_field)


class Consumer(object):
    q = Queue(maxsize=10000)
    p = queue.Queue(maxsize=10000)

    @staticmethod
    def df(x):
        return x + 1

    @classmethod
    @gen.coroutine
    def abc(cls):
        item = yield cls.q.get()
        item += 1
        return cls.df(item)

    @classmethod
    @gen.coroutine
    def consume(cls):
        log.info("[biz] Consumer: i am sleeping")
        time.sleep(3)
        log.info("[biz] Consumer: wake up")
        while True:
            item = yield cls.abc()
            string = 'Doing work on %s, time %s' % (item, time.time())
            cls.p.put(string)


class Sync_io(object):
    def __init__(self):
        self.pool_write = ThreadPoolExecutor(max_workers=1)
        self.pool_read = ThreadPoolExecutor(max_workers=1)

    def write(self, *args):
        """

        :param args: (path, write_flag, data)
        :return:
        """
        # self.with_write(*args)
        # return 0
        self.pool_write.submit(self.with_write, *args)
        return 0

    def read(self, *args):
        """
        pool.submit do not return result, use map instead
        :param args: (path, read_start, read_nrows)
        :return:
        """
        # return self.with_read(*args)
        tmp = self.pool_read.submit(self.with_read, *args)
        return tmp.result()

    @staticmethod
    def with_write(path, write_flag, data):
        with open(path, write_flag, encoding="utf8") as f:
            f.writelines(data)
        return 0

    @staticmethod
    def with_read(path, read_start, read_nrows, pointer=False):
        with open(path, "r", encoding="utf8") as f:
            line_read = LineRead(f)
            if pointer:
                data = line_read.read(read_start, read_nrows)
            else:
                data = line_read.read_line(read_start, read_nrows)
        tmp = json.dumps({"data": data})
        return tmp


def understand_data(data):
    if isinstance(type(data), bytes):
        try:
            data = pickle.loads(data)
        except:
            raise ValueError("Can not pickle bytes!")
    else:
        try:
            data = json.loads(data)
        except:
            pass
    return data


def linenum_to_pointer(path, line_num):
    pointer_lst = [0]
    with open(path, "rb") as f:
        data = f.read()
    for k, piece in enumerate(re.finditer(b"\n", data)):
        if (k + 1) % line_num == 0:
            pointer_lst.append(piece.span()[1])
    if len(data[pointer_lst[-1]:]) == 0:
        return pointer_lst[:-1]
    else:
        return pointer_lst


class LineRead(object):
    def __init__(self, f):
        self.here = 0
        self.f = f
        self.eof = False

    def read(self, here, line_num):
        """
        f = open("../translate.log", "r", encoding="utf8")
        line_read = LineRead(f)
        data, pos = line_read.read(500, 0)
        data, pos = line_read.read(500, pos)
        data, pos = line_read.read(500, pos)
        data, pos = line_read.read(500, pos)
        len(data) == 446
        """
        self.here = here
        self.f.seek(self.here)
        data = []
        for i in range(line_num):
            line = self.f.readline()
            if line == "":
                self.eof = True
                break
            data.append(line)
        self.here = self.f.tell()
        return data

    def read_line(self, read_start, read_nrows):
        """

        :param f: file flow
        :param read_start: start line
        :param read_nrows: num of rows
        :return:
        """
        return [line for k, line in enumerate(self.f) if read_start <= k < read_start + read_nrows]


if __name__ == '__main__':
    sync = Sync_io()
    tmp = sync.write("try2.txt", "w", ["dsd\n", "dsadsda\n", "dasd\n"])
    print(tmp)
    pool = ThreadPoolExecutor(max_workers=4)


    def addd(x, k):
        return x + k


    num = pool.map(lambda args: addd(*args), zip([1, 2], [300, 400]))
    print(list(num))

    print(sync.read("try2.txt", 0, 2))
