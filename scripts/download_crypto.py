#!/usr/bin/env python3
import requests, time, os, json
from datetime import datetime as dt

SYMBOLS = ['BTCUSDT','ETHUSDT','BNBUSDT','XRPUSDT','ADAUSDT','DOGEUSDT','SOLUSDT','DOTUSDT','MATICUSDT','LTCUSDT','AVAXUSDT','LINKUSDT','UNIUSDT','ATOMUSDT']

STARTS = {
    'BTCUSDT': 1502937600000, 'ETHUSDT': 1502937600000, 'BNBUSDT': 1510012800000,
    'XRPUSDT': 1525363200000, 'ADAUSDT': 1523923200000, 'DOGEUSDT': 1561939200000,
    'SOLUSDT': 1597180800000, 'DOTUSDT': 1597795200000, 'MATICUSDT': 1556668800000,
    'LTCUSDT': 1513123200000, 'AVAXUSDT': 1601078400000, 'LINKUSDT': 1547760000000,
    'UNIUSDT': 1600128000000, 'ATOMUSDT': 1556668800000,
}
END_TS = 1640995200000
OUT_DIR = '/Users/mark/Git/hub/mark-tsf/input/Crypto'
PFILE = '/tmp/crypto_progress2.json'
URL = 'https://api.binance.com/api/v3/klines'

def load_p():
    if os.path.exists(PFILE):
        return json.load(open(PFILE))
    return {}

def save_p(sym, st):
    p = load_p(); p[sym] = st; json.dump(p, open(PFILE,'w'))

def valid(sym):
    p = os.path.join(OUT_DIR, sym+'.csv')
    if not os.path.exists(p): return False
    with open(p) as f: return sum(1 for _ in f) > 1

def fetch(sym, start, end, resume=0):
    out = os.path.join(OUT_DIR, sym+'.csv')
    os.makedirs(OUT_DIR, exist_ok=True)
    cur = start; rows = resume; batch = resume//1000
    if resume == 0:
        with open(out,'w') as f: f.write('open_time,open,high,low,close,volume\n')
    while cur < end:
        try:
            r = requests.get(URL, params={'symbol':sym,'interval':'1m','startTime':cur,'endTime':end,'limit':1000}, timeout=20)
            if r.status_code == 200 and r.json():
                data = r.json()
                with open(out,'a') as f:
                    for row in data: f.write('%s,%s,%s,%s,%s,%s\n' % (row[0],row[1],row[2],row[3],row[4],row[5]))
                rows += len(data); cur = data[-1][6]+1
                if batch % 50 == 0: print('  [%s] batch %d: %d rows, %s' % (sym,batch,rows,dt.fromtimestamp(cur/1000).strftime('%Y-%m-%d')), flush=True)
                batch += 1
            else: time.sleep(1)
        except Exception as e:
            print('  [%s] %s' % (sym, e), flush=True); time.sleep(2)
        time.sleep(0.05)
    save_p(sym, {'complete': True, 'rows': rows})
    return rows

if __name__ == '__main__':
    p = load_p()
    done = [s for s in SYMBOLS if p.get(s,{}).get('complete') and valid(s)]
    print('Already done: %d/%d' % (len(done), len(SYMBOLS)))
    for sym in SYMBOLS:
        if sym in done: continue
        resume = p.get(sym,{}).get('rows',0) if valid(sym) else 0
        print('[%s] Downloading (resume=%d)...' % (sym, resume), flush=True)
        rows = fetch(sym, STARTS.get(sym,1514764800000), END_TS, resume)
        print('  %s: %d rows' % (sym, rows))
    print()
    for f in sorted(os.listdir(OUT_DIR)):
        if f.endswith('.csv'):
            sz = os.path.getsize(os.path.join(OUT_DIR, f))
            print('  %s: %.0f MB' % (f, sz/1024**2))
