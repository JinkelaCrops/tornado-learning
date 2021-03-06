"""
为了能够按行读取，选定区间读取，以及并发写入，需要以下的修改
1. RemoteIO.save 写入方式可选
2. RemoteIO.laod 读取起始行及行数可选

file-down的目标文件夹可选
"""

import requests
import json
from concurrent.futures import ThreadPoolExecutor


class RemoteIO(object):
    def __init__(self, host):
        self.host = host

    def save(self, data, file_name, write_mode):
        payload = {
            "data": data,
            "file_name": file_name,
            "write_mode": write_mode
        }
        requests.post(url="http://%s:8000/wt" % self.host, data=payload)

    def load(self, file_name, read_start, read_size, pointer=0):
        payload = {
            "file_name": file_name,
            "read_start": read_start,
            "read_nrows": read_size,
            "pointer": pointer,
        }
        res = requests.post(url="http://%s:8000/sd" % self.host, data=payload)
        return res.json()["data"]

    def exec(self, python_code, result_name):
        payload = {
            "python_code": python_code,
            "result_name": result_name
        }
        res = requests.post(url="http://%s:8000/ex" % self.host, data=payload)
        return res.json()["result"]

    def test1(self, args):
        payload = {
            "args": args
        }
        requests.post(url="http://%s:8000/test1" % self.host, data=payload)

    def test2(self):
        res = requests.post(url="http://%s:8000/test2" % self.host)
        return res.json()["result"]

    def trans_send(self, args):
        payload = {
            "args": args
        }
        requests.post(url="http://%s:8000/trans_send" % self.host, data=payload)

    def trans_get(self):
        res = requests.post(url="http://%s:8000/trans_get" % self.host)
        return res.json()["result"]

    def trans2(self, args):
        payload = {
            "args": json.dumps(args)
        }
        executer = ThreadPoolExecutor(max_workers=2)
        a = executer.submit(requests.post, url="http://%s:8000/trans" % self.host, data=payload)
        b = executer.submit(requests.post, url="http://%s:8001/trans" % self.host, data=payload)
        return 1

    def trans(self, args):
        payload = {
            "args": json.dumps(args)
        }
        res = requests.post(url="http://%s:8000/trans" % self.host, data=payload)
        return res.json()["result"]