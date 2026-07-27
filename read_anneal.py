import re, glob, sys

pat = re.compile(r'ANNEAL epoch (\d+)\s+d=([\d.]+)\s+ep_cost=([\d.]+)\s+lambda=([\d.eE+-]+)')
ref = {21:1.10, 22:1.10, 23:1.12, 24:1.13, 25:1.15,
       26:1.18, 27:1.22, 28:1.28, 29:1.39, 30:1.65}

files = sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1
                         else 'slurm_logs/tipanneal_*.out'))
if not files:
    raise SystemExit('no logs matched')

for f in files:
    print(f.split('/')[-1])
    seen = False
    for line in open(f):
        m = pat.search(line)
        if not m:
            continue
        seen = True
        e, d = int(m.group(1)), float(m.group(2))
        c, lam = float(m.group(3)), float(m.group(4))
        B = c / d if d else float('nan')
        r = ref.get(e)
        flag = ''
        if B > 1.7:
            flag = '  ABORT'
        elif r is not None and B > r + 0.3:
            flag = '  LAGGING'
        print('  e=%2d  d=%7.2f  cost=%8.2f  B=%5.2f  lambda=%.6f%s'
              % (e, d, c, B, lam, flag))
    if not seen:
        print('  no ANNEAL lines yet, still in warmup')
