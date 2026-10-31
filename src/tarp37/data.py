from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
import json
import numpy as np
from .assets import PROJECT
from .contract import PhysicalContract


class ForecastTrace:
    def __init__(self, planning_ev_mwh_per_day=57.663, renewable_share=.7):
        data = np.load(PROJECT/'artifacts/data/hourly.npz')
        self.times = data['time_s']; original=data['actual']
        self.raw=original.copy(); self.forecasts = data['forecast']; self.issues = data['issues'].copy()
        # Immutable legacy export joined EIA period labels directly to CAISO
        # interval starts. EIA WND/SUN are HOUR ENDING: period T+1 describes
        # [T,T+1), whereas CAISO ACTUAL load already uses interval start T.
        # Reindex only actual renewable columns; never shift load, prices or
        # issued forecast vintages. A missing following hour stays unknown.
        end=np.searchsorted(self.times,self.times+3600)
        in_range=end<len(self.times)
        valid=np.zeros(len(end),bool)
        valid[in_range]=self.times[end[in_range]]==self.times[in_range]+3600
        self.raw[:,1:3]=np.nan
        self.raw[valid,1:3]=original[end[valid],1:3]
        ref = json.loads((PROJECT/'artifacts/data/reference.json').read_text(encoding='utf-8'))
        c = PhysicalContract.load()
        base = np.mean([sum(p for p,q in c.case.base_load(h*3600).values()) for h in range(24)])
        planning_mean = base + planning_ev_mwh_per_day/24
        desired = planning_mean*renewable_share
        self.scale = np.array([base/ref['mean_load_raw'], desired*.3/ref['mean_wind_raw'], desired*.7/ref['mean_solar_raw'], 1.])
        self.wind_nameplate = self.scale[1]*ref['wind_reference_nameplate_mw']
        self.solar_nameplate = self.scale[2]*ref['solar_reference_nameplate_mw']
        self.reference = dict(base_mean_mw=float(base), planning_mean_mw=float(planning_mean), planning_re_share=renewable_share,
                              planning_ev_mwh_per_day=planning_ev_mwh_per_day, solar_nameplate_mw=self.solar_nameplate,
                              wind_nameplate_mw=self.wind_nameplate, scales=self.scale.tolist(),
                              actual_time_alignment='CAISO load interval start; EIA WND/SUN hour-ending shifted -1h; raw cache unchanged',
                              forecast_time_alignment='original publication-constrained CAISO vintages unchanged',
                              renewable_scale_reference='frozen common development-hour reference; no retuning on audit outcomes')

    def position(self, now):
        i = int(np.searchsorted(self.times, int(now)))
        if i >= len(self.times) or self.times[i] != int(now): raise ValueError('Missing actual/forecast hour')
        return i

    @staticmethod
    def time_for_date(date, local_hour=0):
        return int((datetime.fromisoformat(date).replace(tzinfo=ZoneInfo('America/Los_Angeles'))+timedelta(hours=local_hour)).timestamp())

    def _bounded(self, values):
        x = np.asarray(values, float)*self.scale
        x[...,0] = np.maximum(x[...,0],0)
        x[...,1] = np.clip(x[...,1],0,self.wind_nameplate)
        x[...,2] = np.clip(x[...,2],0,self.solar_nameplate)
        return x

    def actual(self, now):
        x = self.raw[self.position(now)].copy()
        if not np.isfinite(x[:3]).all(): raise ValueError('Missing actual physical observation')
        if not np.isfinite(x[3]): x[3] = 30.
        return self._bounded(x)

    def visible(self, now, horizon=6):
        i = self.position(now); x = self.forecasts[i,:horizon].copy(); issues = self.issues[i,:horizon]
        if not np.isfinite(x[:,:3]).all() or np.any(issues<=0) or np.any(issues>now):
            raise ValueError('Forecast publication/value unavailable')
        # A target day-ahead schedule is only available after its declared release.
        for k in range(horizon):
            local = datetime.fromtimestamp(now+k*3600, ZoneInfo('America/Los_Angeles'))
            release = local.replace(hour=0,minute=0,second=0,microsecond=0)-timedelta(days=1)+timedelta(hours=13)
            if release.timestamp()>now or not np.isfinite(x[k,3]): x[k,3] = 30.
        return dict(now=int(now), values=self._bounded(x), issued_at=issues.copy(),
                    price_issue_assumption='13:00 Pacific previous day, otherwise fixed 30 fallback')
