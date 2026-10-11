"""Read fresh D1 output from the actual read-only fleet-diskcheck service.

The proof service executes fleet-diskcheck once per connection. It uses the
same fixed CPU output declaration that the native CLI proof supplies.
No test response, saved result or synthetic disk value exists in this client.
"""
import argparse
import json
import socket
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--address', required=True)
parser.add_argument('--port', type=int, required=True)
parser.add_argument('--need-gb', type=float, required=True)
parser.add_argument('--hosts', required=True)
parser.add_argument('--paths', required=True)
options = parser.parse_args()
if (options.need_gb != 0.1 or options.hosts != 'dl380g10'
        or options.paths != '/tmp,/mnt/shared'):
    raise ValueError('The read-only CPU disk service has a different output declaration.')
with socket.create_connection((options.address, options.port), timeout=120) as stream:
    data = bytearray()
    while chunk := stream.recv(65536):
        data.extend(chunk)
body = json.loads(data)
print(json.dumps(body), flush=True)
sys.exit(0 if body.get('pass') is True else 1)
