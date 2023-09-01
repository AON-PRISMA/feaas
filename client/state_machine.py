import logging
import os
import threading
from typing import Union

import eventlet
import flask_socketio
import socketio

from crypto_utils import key_derive, encrypt, lsh, gen_non_collision_hash

logger = logging.getLogger('main')

event_has_records = "when it has a record to match"
event_no_records = "no more records"
event_m2 = "m2 received"
event_m6 = "m6 received"
event_m3 = "m3 received"
event_m8 = "m8 received"
event_m4 = "m4 received"
event_send_all = "send all records"
event_null = "null"

s0_id = 0
s1_id = 1
s2_id = 2
s3_id = 3
s4_id = 4
s5_id = 5

DOMAIN_SIZE = 8  # The domain size of the encoding values (default: [0,7])


class StateMachine:
    def __init__(self, record, sio_to_server: socketio.Client, sio_to_clients: Union[socketio.Client, flask_socketio.SocketIO],
                 args):
        self.args = args
        self.currentState = S0()
        self.event_flags = {event_has_records: False,
                            event_no_records: False,
                            event_m2: False,
                            event_m3: False,
                            event_m4: False,
                            event_m6: False,
                            event_m8: False,
                            event_send_all: True,
                            event_null: True}
        self.sio_to_server = sio_to_server
        self.sio_to_clients = sio_to_clients
        self.record = record
        self.dim = len(record[0]['data_record'])  # assume all data records are at the same dimension
        self.last_matching_results = None
        self.key_id = 0
        self.client_id = -1

        self.master_key = None  # a master key agreed during key exchange
        self.thread_lock = threading.Lock()

    def start(self):
        while True:
            self.currentState = self.currentState.next(state_machine=self)
            logger.debug(f'Client ID: {self.client_id}, State ID: {self.currentState.id}')
            self.currentState.run(state_machine=self)
            if self.currentState.id == s0_id:
                return
            eventlet.sleep(0)

    def generate_master_key(self):
        self.master_key = os.urandom(32)  # 32-byte master key

    def set_master_key(self, key):
        self.master_key = key


class State:
    def __init__(self):
        self.id = -1
        self.edges = None

    def run(self, state_machine: StateMachine):
        assert 0, "run not implemented"

    def next(self, state_machine: StateMachine):
        while True:
            with state_machine.thread_lock:  # lock when reading from state_machine.event_flags
                for k, v in self.edges.items():
                    if state_machine.event_flags[k]:
                        return state_map[v]
            eventlet.sleep(0.1)


class S0(State):
    def __init__(self):
        super().__init__()
        self.id = 0
        self.edges = {event_has_records: s1_id}

    def run(self, state_machine):
        pass


class S1(State):
    def __init__(self):
        super().__init__()
        self.id = 1
        self.edges = {event_m2: s2_id, event_m6: s3_id}

    def run(self, state_machine):
        state_machine.event_flags[event_has_records] = False
        if state_machine.args.cid == 1:
            msg = {'id': 1, 'desc': 'msg #1', 'content': {'th': state_machine.args.th}}
        else:
            msg = {'id': 1, 'desc': 'msg #1'}
        state_machine.sio_to_server.emit('message', msg)


class S2(State):
    def __init__(self):
        super().__init__()
        self.id = 2
        self.edges = {event_null: s3_id}

    def run(self, state_machine):
        with state_machine.thread_lock:
            state_machine.event_flags[event_m2] = False
        msg = {'id': 6, 'desc': 'msg #6', 'content': state_machine.master_key}
        state_machine.sio_to_clients.emit('message', msg)


class S3(State):
    def __init__(self):
        super().__init__()
        self.id = 3
        self.edges = {event_m8: s4_id}

    def run(self, state_machine):
        with state_machine.thread_lock:
            state_machine.event_flags[event_m3] = False
            state_machine.event_flags[event_m6] = False
        msg = {'id': 7, 'desc': 'msg #7'}
        state_machine.sio_to_server.emit('message', msg)


class S4(State):
    def __init__(self):
        super().__init__()
        self.id = 4
        self.edges = {event_m3: s3_id, event_no_records: s5_id}

    def run(self, state_machine):
        send_record(state_machine)
        with state_machine.thread_lock:
            state_machine.event_flags[event_no_records] = True
            state_machine.event_flags[event_m3] = False  # this is needed since m3 can be received in s3
            state_machine.event_flags[event_m8] = False
            state_machine.event_flags[event_send_all] = True


class S5(State):
    def __init__(self):
        super().__init__()
        self.id = 5
        self.edges = {event_m4: s0_id, event_m3: s3_id, event_m8: s4_id}

    def run(self, state_machine):
        with state_machine.thread_lock:
            state_machine.event_flags[event_no_records] = False


s0, s1, s2, s3, s4, s5 = S0(), S1(), S2(), S3(), S4(), S5()
state_map = {0: s0, 1: s1, 2: s2, 3: s3, 4: s4, 5: s5}


def send_record(s):
    with s.thread_lock:
        if not s.event_flags[event_send_all] and s.last_matching_results and \
                    s.last_matching_results[0]['streaming_client_id'] == s.client_id:
            matched_row_ind = s.last_matching_results[0]['streaming_client_ind']
        else:
            matched_row_ind = -1

        # perform the key derivation here to ensure the key_id is up-to-date
        sub_keys, derived_key, perm_idx = key_derive(s.master_key, s.key_id, s.dim, s.args.rep)
        poly_degree = s.dim - 2 * s.args.th

        record_copy = {}
        for k in s.record:
            if k <= matched_row_ind:
                continue
            if s.args.plain:
                enc_record = s.record[k]['data_record']
            else:
                hash_table = gen_non_collision_hash(sub_keys, list(range(DOMAIN_SIZE)), s.args.num_bits)
                enc_record = encrypt(s.record[k]['data_record'], hash_table, poly_degree, s.args.rep, s.args.num_bits, perm_idx)
            lsh_vals = lsh(s.record[k]['data_record'], s.args.lsh_rep, s.args.lsh_num_ind, s.args.lsh_num_bin, derived_key)
            record_copy[k] = {'data_record': enc_record, 'lsh_val': lsh_vals, 'key_id': s.key_id}
        record_copy[list(s.record.keys())[-1]]['last_record'] = True
        logger.debug(f"Len of sent data: {len(record_copy)}")
        s.sio_to_server.emit('data_record', record_copy)
