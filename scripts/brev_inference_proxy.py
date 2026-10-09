"""Loopback-only replica routing; request and response bodies are not modified."""
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time


def response_problem(payload, response):
    try:
        choice = response['choices'][0]
        if response['model'] != payload['model'] or len(response['choices']) != 1:
            return 'model_or_choice_count'
        ids = choice.get('token_ids')
        if not isinstance(ids,list) or not all(type(t) is int for t in ids):
            return 'missing_token_ids'
        if choice['finish_reason'] not in ('stop','length'):
            return 'aborted_or_unexpected_finish_reason'
        if (response['usage']['prompt_tokens'] != len(payload['prompt']) or
                response['usage']['completion_tokens'] != len(ids)):
            return 'token_accounting'
    except (KeyError,TypeError,IndexError):
        return 'malformed_response'
    return None


class ReplicaProxy(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self,port,replicas,audit):
        if not replicas or any(type(r['local_port']) is not int for r in replicas):
            raise ValueError('Expected explicit loopback replica ports')
        self.replicas = replicas
        self.audit = audit
        self.cursor = 0
        self.routing_lock = threading.Lock()
        super().__init__(('127.0.0.1',port),Handler)

    def choose(self,completion):
        with self.routing_lock:
            selected = self.replicas[self.cursor % len(self.replicas)] if completion else self.replicas[0]
            if completion: self.cursor += 1
            return selected


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self,*args):
        pass

    def do_GET(self):
        self.forward(False)

    def do_POST(self):
        self.forward(True)

    def forward(self,completion):
        expected = '/v1/completions' if completion else '/v1/models'
        if self.path != expected:
            self.send_error(404); return
        size = int(self.headers.get('Content-Length','0'))
        if size < 0 or size > 8*1024*1024 or self.headers.get('Transfer-Encoding'):
            self.send_error(400); return
        body = self.rfile.read(size) if completion else None
        replica = self.server.choose(completion)
        started = time.monotonic()
        connection = http.client.HTTPConnection('127.0.0.1',replica['local_port'],timeout=3700)
        problem = None
        try:
            connection.request('POST' if completion else 'GET',self.path,body,
                               {'Content-Type':'application/json'})
            result = connection.getresponse()
            status,raw = result.status,result.read()
            if completion:
                try:
                    problem = response_problem(json.loads(body),json.loads(raw)) if status == 200 else 'http_error'
                except (ValueError,UnicodeError):
                    problem = 'non_json_response'
        except (OSError,http.client.HTTPException) as error:
            status = 502
            raw = json.dumps({'error':'Inference transport failed','type':type(error).__name__}).encode()
            problem = 'transport_error'
        finally:
            connection.close()
        if completion:
            self.server.audit(dict(replica=replica['name'],gpu_id=replica['gpu_id'],
                request_sha256=hashlib.sha256(body).hexdigest(),http_status=status,
                elapsed_seconds=time.monotonic()-started,problem=problem),
                body if problem else None,raw if problem else None)
        self.send_response(status)
        self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(raw)))
        self.end_headers()
        try: self.wfile.write(raw)
        except (BrokenPipeError,ConnectionResetError): pass
