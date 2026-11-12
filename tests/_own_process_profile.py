"""Profile a pytest child without swallowing its terminal exit status."""
import cProfile
import runpy
import sys


def main() -> None:
    output = sys.argv.pop(1)
    profiler = cProfile.Profile()
    try:
        profiler.runcall(runpy.run_module, "pytest", run_name="__main__",
                        alter_sys=True)
    finally:
        profiler.dump_stats(output)


if __name__ == "__main__":
    main()
