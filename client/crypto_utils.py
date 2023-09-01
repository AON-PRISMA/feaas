import ctypes
import hmac
import os
from ctypes import POINTER, c_uint16, c_uint32

import numpy as np
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

HASH_ADDITIVE_CONSTANT = 42


# only works for non-negative int
def int_to_bytes(x: int) -> bytes:
    return x.to_bytes((x.bit_length() + 7) // 8, 'big')


def int_from_bytes(x_bytes: bytes) -> int:
    return int.from_bytes(x_bytes, 'big')


def key_derive(master_key, key_id, length, rep):
    derived_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=int_to_bytes(key_id),
    ).derive(master_key)

    np.random.seed(int_from_bytes(derived_key) % (2 ** 32))  # take a mod
    idx_list = np.random.permutation(length)

    sub_keys = []
    for i in range(length * rep):
        key_i = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=int_to_bytes(i),
        ).derive(derived_key)
        sub_keys.append(key_i)
    return sub_keys, derived_key, idx_list


def encrypt(data, hash_table, poly_degree, rep, num_bits, perm_idx=None):
    cur_file_dir = os.path.dirname(os.path.realpath(__file__))
    so = ctypes.cdll.LoadLibrary(os.path.join(cur_file_dir, 'share_gen/_share_gen.so'))
    gen_shares = so.generateShares
    free_ptr = so.freePtr
    if num_bits == 16:
        gen_shares.restype = POINTER(c_uint16)
        free_ptr.argtypes = (ctypes.POINTER(c_uint16), )
    elif num_bits == 32:
        gen_shares.restype = POINTER(c_uint32)
        free_ptr.argtypes = (ctypes.POINTER(c_uint32), )
    else:
        raise NotImplementedError

    shared_poly_ptr = gen_shares(len(data) * rep, poly_degree)
    shared_poly = [shared_poly_ptr[i] for i in range(len(data) * rep)]
    free_ptr(shared_poly_ptr)  # free memory to avoid memory leak

    if perm_idx is None:
        perm_idx = np.arange(len(data))
    encrypted_data = []
    for i in range(len(data) * rep):
        idx = perm_idx[i // rep]
        encrypted_data.append(hash_table[i][data[idx]] ^ shared_poly[i])

    return encrypted_data


def gen_non_collision_hash(sub_keys, domain, num_bits):
    ret_table = [{} for _ in range(len(sub_keys))]
    for i in range(len(sub_keys)):
        for x in domain:
            # take the first 2 bytes for 16 bit, 4 for 32 bits
            h = int_from_bytes(hmac.digest(sub_keys[i], int_to_bytes(x), digest='sha256')[:num_bits//8])
            counter = 1
            while h in list(ret_table[i].values()):
                h = int_from_bytes(hmac.digest(sub_keys[i], int_to_bytes(x + HASH_ADDITIVE_CONSTANT * counter), digest='sha256')[:num_bits//8])
                counter += 1
            ret_table[i][x] = h
    return ret_table


def lsh(data, repetition, num_ind, num_bin, derived_key):
    ret = []
    np.random.seed(int_from_bytes(derived_key) % (2 ** 32))  # set the seed to be derived_key
    for _ in range(repetition):
        indices = np.random.permutation(len(data))[:num_ind]
        byte_data = bytes([data[i] for i in indices])  # assume 0 <= data[i] <256
        hash_val = hmac.digest(derived_key, byte_data, digest='sha256')
        hash_val = int_from_bytes(hash_val) % num_bin
        ret.append(hash_val)
    return ret
