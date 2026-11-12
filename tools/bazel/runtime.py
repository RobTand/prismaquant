"""Give a remote test the environment the hosted gate gives a shard."""
import os
from pathlib import Path
import sys
import tempfile
import venv

# .github/workflows/gate.yml and tools/ci/gate_run.py set these for every shard.
GATE_ENVIRONMENT = {
    'OMP_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1',
    'NUMEXPR_NUM_THREADS': '1',
    'TORCHINDUCTOR_COMPILE_THREADS': '1',
    'MAX_JOBS': '1',
    'CUDA_VISIBLE_DEVICES': '',
    'GIT_AUTHOR_NAME': 'PrismaQuant CI',
    'GIT_AUTHOR_EMAIL': 'ci@example.invalid',
    'GIT_COMMITTER_NAME': 'PrismaQuant CI',
    'GIT_COMMITTER_EMAIL': 'ci@example.invalid',
}
# What Bazel's test runner adds. A hosted shard has none of it.
BAZEL_PREFIXES = ('TEST_', 'RUNFILES_', 'BASH_FUNC_', 'GTEST_', 'COVERAGE_', 'TESTBRIDGE_', 'LD_')
BAZEL_NAMES = {'JAVA_RUNFILES', 'PYTHON_RUNFILES', 'XML_OUTPUT_FILE', 'BAZEL_TEST', 'PYTHONSAFEPATH',
               'RUNTEST_PRESERVE_CWD', 'TZ', 'BUILD_EXECROOT', 'PQ_BAZEL_LANE', 'PQ_BAZEL_BUDGETS'}


def secret(name):
    """A credential the Remote Bazel runner VM exports (BUILDBUDDY_API_KEY) or any
    *API_KEY*/*TOKEN*: no test needs one, and a failure that prints its
    environment would write it into an uploaded log."""
    upper = name.upper()
    return 'API_KEY' in upper or 'TOKEN' in upper


def gate_environment(environ, root, scratch):
    """The pytest process's environment: the caller's minus Bazel's, plus the gate's.

    gate_run also drops loader overrides (LD_*) so proof capture sees the real
    interpreter. TMPDIR and HOME are short and off /tmp: hosted shards use
    $RUNNER_TEMP, some tests refuse a work root under /tmp, and Bazel's
    TEST_TMPDIR is too long for the AF_UNIX sockets tests bind.
    """
    environment = {name: value for name, value in environ.items()
                   if not name.startswith(BAZEL_PREFIXES) and name not in BAZEL_NAMES and not secret(name)}
    environment.update(GATE_ENVIRONMENT)
    environment['TMPDIR'] = environment['HOME'] = str(scratch)
    # gate_run puts tools/ci first; the gate workflow sets PYTHONPATH=. at the root.
    environment['PYTHONPATH'] = os.pathsep.join([str(root / 'tools/ci'), str(root)])
    return environment


def short_scratch(private):
    """A new directory as near the root as possible, on TEST_TMPDIR's filesystem.

    A remote executor's /tmp is overlay, and TEST_TMPDIR is too deep for
    AF_UNIX sockets; its filesystem's top writable directory is neither.
    """
    device, base = private.stat().st_dev, private
    for parent in private.parents:
        if parent.stat().st_dev != device or not os.access(parent, os.W_OK | os.X_OK):
            break
        base = parent
    return Path(tempfile.mkdtemp(prefix='t', dir=base))


def checkout_of(runfiles):
    """The runner's git checkout: runfiles of a local, unsandboxed test link into it."""
    root = (runfiles / 'pytest.ini').resolve().parent
    if not (root / '.git').exists():
        raise RuntimeError(f'runner-lane test needs a git checkout; {root} is not one. '
                           'Run it on a Remote Bazel runner (docs/bazel.md).')
    return root


def prepare_remote_test(root):
    """Short scratch and an interpreter whose children keep Bazel's packages.

    Returns the scratch directory and the environment for pytest.
    """
    scratch = short_scratch(Path(os.environ['TEST_TMPDIR']).resolve())
    # Python's -I ignores PYTHONPATH. A real venv with a .pth keeps declared
    # Bazel dependencies for children that clear their environment, as the
    # hosted gate's editable install keeps prismaquant importable everywhere.
    environment = Path(os.environ['TEST_TMPDIR']).resolve() / 'python-environment'
    if not (environment / 'pyvenv.cfg').exists():
        venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
    site = next(environment.glob('lib/python*/site-packages'))
    # The installed packages, not the checkout: the hosted gate's editable
    # install puts only the prismaquant package on every interpreter's path,
    # so a child in another directory doesn't import experiments or tools.
    paths = [str(Path(path).resolve()) for path in sys.path
             if path and Path(path).resolve() not in (root, Path.cwd().resolve())]
    (site / 'bazel-runfiles.pth').write_text('\n'.join(paths) + '\n')
    package = site / 'prismaquant'
    if not package.exists():
        package.symlink_to(root / 'prismaquant', target_is_directory=True)
    sys.executable = str(environment / 'bin/python')
    # A pip install puts console scripts beside the interpreter; so do these.
    for script in sorted(Path('..').glob('*pq_pip*/bin/*')):
        link = environment / 'bin' / script.name
        if not link.exists():
            link.symlink_to(script.resolve())
    variables = gate_environment(os.environ, root, scratch)
    # Tests run python3, python and those scripts from PATH.
    variables['PATH'] = os.pathsep.join([str(environment / 'bin'), os.environ.get('PATH', '/usr/bin:/bin')])
    return scratch, variables
