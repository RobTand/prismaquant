#!/usr/bin/env python3
"""G3 launch argv adapter, not a numerical scorer."""
import sys
import stage1
if __name__ == '__main__':
    args = sys.argv[1:]
    if '--manifest-sha256' in args:
        i = args.index('--manifest-sha256')
        manifest = args[args.index('--manifest')+1]
        stage1.require(stage1.sha(manifest) == args[i+1], 'Manifest own-file integrity')
        del args[i:i+2]
    sys.argv = [sys.argv[0], 'score', *args]
    stage1.main()
