"""Use the existing Parquet-capable runtime once; simulation needs only NPZ."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
source = root.parent/'tarp_x_critical/.worktrees/v2sim-critical-td3/src/tarp_x_critical/data/processed/real_calibrated_15m_v1/temporal_15m.parquet'
out = root/'artifacts/data'


def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


if out.exists():
    m = json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    for p, h in {**m['inputs'], **m['outputs']}.items():
        if sha(p) != h: raise ValueError('Existing prepared data differs; retain it and create a new data version')
    print('Existing immutable forecast export verified')
else:
    frame = pd.read_parquet(source)
    frame['event_time_utc'] = pd.to_datetime(frame['event_time_utc'], utc=True)
    frame = frame[frame.event_time_utc.dt.minute == 0].sort_values('event_time_utc')
    t = frame.event_time_utc.astype('int64').to_numpy()/1e9
    actual = frame[['load_actual', 'wind_actual', 'solar_actual', 'real_time_price']].to_numpy(float)
    forecast = np.empty((len(frame), 6, 4)); issues = np.empty((len(frame), 6, 3))
    for j, h in enumerate([1,4,8,12,16,20]):
        for k, name in enumerate(['load','wind','solar']):
            forecast[:,j,k] = frame[f'{name}_forecast_h{h:02}'].to_numpy(float)
            issues[:,j,k] = pd.to_datetime(frame[f'{name}_forecast_issue_h{h:02}'],utc=True).astype('int64').to_numpy()/1e9
        # Day-ahead schedule is joined from the target hour; release assumption
        # is explicitly checked by the reader, never real-time future prices.
        targets = frame.event_time_utc+pd.Timedelta(hours=j)
        prices = frame.set_index('event_time_utc')['day_ahead_price'].reindex(targets).to_numpy(float)
        forecast[:,j,3] = prices
    train = frame['split'].eq('train').to_numpy()
    common = train & np.isfinite(actual[:,:3]).all(axis=1)
    capacity_file = root.parent/'tarp_v2sim_nanjing_eia/artifacts/reference_data/capacity_reference.json'
    capacity = json.loads(capacity_file.read_text(encoding='utf-8'))
    reference = dict(mean_load_raw=float(np.mean(actual[common,0])),
                     mean_wind_raw=float(np.mean(np.maximum(actual[common,1],0))),
                     mean_solar_raw=float(np.mean(np.maximum(actual[common,2],0))),
                     reference_common_hours=int(common.sum()),
                     solar_reference_nameplate_mw=capacity['solar_nameplate_mw'],
                     wind_reference_nameplate_mw=capacity['wind_nameplate_mw'])
    out.mkdir(parents=True)
    np.savez_compressed(out/'hourly.npz', time_s=t.astype('int64'), actual=actual,
                        forecast=forecast, issues=issues, role=frame['split'].to_numpy(dtype='U16'))
    (out/'reference.json').write_text(json.dumps(reference,indent=2),encoding='utf-8')
    inputs={str(source):sha(source),str(Path(__file__)):sha(__file__),str(capacity_file):sha(capacity_file)}
    (out/'manifest.json').write_text(json.dumps(dict(inputs=inputs,outputs={str(p):sha(p) for p in out.iterdir() if p.is_file()},
        forecast_provenance='Existing publication-constrained forecast columns; no new EIA forecast fields invented',
        price_publication='day-ahead schedule released 13:00 Pacific previous day assumption'),ensure_ascii=False,indent=2),encoding='utf-8')
    print(dict(rows=len(frame), output=str(out), **reference))
