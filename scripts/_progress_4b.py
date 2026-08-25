import datetime
import re

LOG = "logs/scifact_official_frontier_max_4b.log"
lines = open(LOG, errors="ignore").read().splitlines()
start = max(
    i for i, l in enumerate(lines) if "23:28:10" in l and "official SciFact start" in l
)
ts = []
for l in lines[start:]:
    m = re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ httpx.*200 OK", l)
    if m:
        ts.append(datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))

gaps = [(ts[i + 1] - ts[i]).total_seconds() for i in range(len(ts) - 1)]
recent = gaps[-20:]
rate = sum(recent) / len(recent)
done = len(ts)
qb, cb = -(-300 // 32), -(-5183 // 32)
per_variant = qb + cb
total = per_variant * 2
rem = (total - done) * rate

print(f"completed batches : {done} / {total}   ({done / total:.0%})")
print(f"  per variant     : {qb} query + {cb} corpus = {per_variant}, two variants")
print(f"recent rate       : {rate:.1f}s per batch of 32 (last 20) -- fully serialized")
print(f"elapsed           : {(ts[-1] - ts[0]).total_seconds() / 60:.0f} min")
eta = ts[-1] + datetime.timedelta(seconds=rem)
print(f"remaining         : {rem / 3600:.1f} h  -> ETA {eta.strftime('%H:%M')} UTC")
