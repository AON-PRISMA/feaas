import copy
import logging

import socketio

import client_coordinator
import state_machine

logger = logging.getLogger('main')


class Client:
    def __init__(self, data, args):
        self.args = args
        self.sio = self.sio_init()
        self.sub_sio = self.sub_sio_init()
        self.client_state_machine = state_machine.StateMachine(data, self.sio, self.sub_sio, args)
        self.client_state_machine.event_flags[state_machine.event_has_records] = True
        if args.cid == 1:
            self.client_state_machine.generate_master_key()

    def run(self):
        self.client_state_machine.start()
        self.sio.disconnect()

    def sio_init(self):
        if self.args.tls_file == '':
            sio = socketio.Client(reconnection=False)
        else:
            sio = socketio.Client(ssl_verify=self.args.tls_file, reconnection=False)

        @sio.event
        def connect():
            logger.debug("I'm connected!")

        @sio.event
        def connect_error(data):
            logger.debug("The connection failed!")

        @sio.event
        def disconnect():
            logger.debug("I'm disconnected!")

        @sio.on('message')
        def handle_result(ret):
            logger.debug(ret)
            with self.client_state_machine.thread_lock:
                if ret['id'] == 2:
                    self.client_state_machine.client_id = ret['content']['clientId']
                    if self.client_state_machine.args.cid == 1:
                        # set to True only for c1, which coordinates the key exchange
                        self.client_state_machine.event_flags[state_machine.event_m2] = True
                elif ret['id'] == 3:
                    logger.info('Match found! Matching info:')

                    # add 1 to record ind so the output record ids count from 1
                    matched_record_ids = copy.deepcopy(ret['content']['recordIds'])
                    for i in range(len(matched_record_ids)):
                        matched_record_ids[i]['streaming_client_ind'] += 1
                        matched_record_ids[i]['matching_client_ind'] += 1
                    logger.info(matched_record_ids)

                    self.client_state_machine.key_id += 1
                    self.client_state_machine.last_matching_results = ret['content']['recordIds']
                    if not ret['content']['endFlag']:
                        self.client_state_machine.event_flags[state_machine.event_m3] = True
                elif ret['id'] == 4:
                    self.client_state_machine.event_flags[state_machine.event_m4] = True
                elif ret['id'] == 8:
                    self.client_state_machine.event_flags[state_machine.event_m8] = True
                    self.client_state_machine.event_flags[state_machine.event_send_all] = ret['content']['sendAll']
                else:
                    logger.info('Incorrect message id from the server.')
                    exit(1)
            return
        if self.args.tls_file == '':
            addr = f'http://{self.args.server_addr}'
        else:
            addr = f'https://{self.args.server_addr}'
        sio.connect(addr)  # connect to the server socket, authentication optional
        return sio

    def sub_sio_init(self):
        if self.args.cid == 1:
            sub_sio = client_coordinator.sio_coordinator
            sub_sio.start_background_task(client_coordinator.coordinator_init, self.args.client_addr)
        else:
            sub_sio = socketio.Client(reconnection=False)

            @sub_sio.on('message')
            def handle_result(ret):
                with self.client_state_machine.thread_lock:
                    if ret['id'] == 6:
                        logger.debug('Received from client 1:', ret)
                        self.client_state_machine.set_master_key(ret['content'])
                        self.client_state_machine.event_flags[state_machine.event_m6] = True
                        logger.debug(f'client {self.args.cid} disconnecting from client 1')
                        sub_sio.disconnect()  # disconnect from client 1
                    else:
                        logger.info('Incorrect message id from other clients.')
                        exit(1)
                return
            # connect to the coord client socket, authentication optional
            sub_sio.connect(f'http://{self.args.client_addr}')
        return sub_sio
