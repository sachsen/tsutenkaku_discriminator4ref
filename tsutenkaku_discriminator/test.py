import pickle


import os.path

file_path = "./data/temp/test.pickle"
n_bytes = 2**33
max_bytes = 2**31 - 1
data = bytearray(n_bytes)

## write
bytes_out = pickle.dumps(data,protocol=pickle.HIGHEST_PROTOCOL)
with open(file_path, 'wb') as f_out:
    for idx in range(0, len(bytes_out), max_bytes):
        f_out.write(bytes_out[idx:idx+max_bytes])

## read
bytes_in = bytearray(0)
input_size = os.path.getsize(file_path)
with open(file_path, 'rb') as f_in:
    for _ in range(0, input_size, max_bytes):
        bytes_in += f_in.read(max_bytes)
data2 = pickle.loads(bytes_in)

assert(data == data2),"aaa"