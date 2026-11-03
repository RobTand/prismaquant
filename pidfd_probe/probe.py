"""Scratch-only runtime capability probe. Prints no process or environment data."""
import json
import os
import platform
import signal
import sys
import sysconfig


def report():
    checks = ((os, 'pidfd_open'), (os, 'P_PIDFD'), (os, 'waitid'),
              (os, 'WNOWAIT'), (signal, 'pidfd_send_signal'))
    capabilities = {name: hasattr(module, name) for module, name in checks}
    result = {'version': sys.version, 'libc': platform.libc_ver(),
              'host_gnu_type': sysconfig.get_config_var('HOST_GNU_TYPE'),
              'capabilities': capabilities}
    if all(capabilities.values()):
        child = os.fork()
        if child == 0:
            signal.pause()
            os._exit(0)
        descriptor = None
        try:
            descriptor = os.pidfd_open(child, 0)
            result['pidfd_open_errno'] = 0
            os.waitid(os.P_PIDFD, descriptor, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            result['waitid_errno'] = 0
            signal.pidfd_send_signal(descriptor, 0)
            result['pidfd_send_signal_errno'] = 0
        except OSError as error:
            result['syscall_errno'] = error.errno
        finally:
            os.kill(child, signal.SIGKILL)
            os.waitpid(child, 0)
            if descriptor is not None:
                os.close(descriptor)
    print('PQ_PIDFD_PROBE ' + json.dumps(result, sort_keys=True), flush=True)
    return all(capabilities.values()) and result.get('pidfd_send_signal_errno') == 0


if __name__ == '__main__':
    raise SystemExit(0 if report() else 1)
