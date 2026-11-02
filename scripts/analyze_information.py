"""Post-query evaluation and scientific report; never used by an acquisition arm."""
import argparse,json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',default='information_pilot_v1');p.add_argument('--native-run',default='v5_information_sanity_v1');a=p.parse_args()
    project=Path(__file__).resolve().parents[1];root=project/'artifacts/information'/a.run_id
    b=read(root/'cache.json');summary=read(root/'summary.json');assert summary['complete']
    arms=[read(root/f'{m}_s{s}.json') for s in [42,43,44] for m in ['random','sobol','error_td3','space_td3']]
    hashes={x['cache_hash'] for x in arms};assert hashes=={sha(root/'cache.json')}
    initial={}
    for r in b['initial']:
        if r['profile'] is not None:initial.setdefault(r['key'],[]).append(np.array(r['profile']))
    # A privileged finite-pool diagnostic, kept separate from all learned arms.
    keys=sorted(initial);remaining={k:[p for p in b['public'] if p['key']==k] for k in keys};known={k:list(v) for k,v in initial.items()};oracle_ids=[]
    for i in range(128):
        k=keys[i%len(keys)];c=remaining[k];scores=[]
        for candidate in c:
            z=b['answers'][candidate['id']]['profile']
            scores.append(-1. if z is None else float(np.linalg.norm(np.array(known[k])-z,axis=1).min()))
        selected=c.pop(int(np.argmax(scores)));oracle_ids.append(selected['id'])
        z=b['answers'][selected['id']]['profile']
        if z is not None:known[k].append(np.array(z))
    oracle_dist=[]
    for k in keys:
        pool=[b['answers'][p['id']]['profile'] for p in b['public'] if p['key']==k and b['answers'][p['id']]['profile'] is not None]
        oracle_dist.append(np.linalg.norm(np.array(pool)[:,None,:]-np.array(known[k])[None,:,:],axis=2).min(1).mean())
    table=[]
    for method in ['random','sobol','error_td3','space_td3']:
        rows=[x['evaluation'][-1] for x in arms if x['method']==method]
        baseline=[x['evaluation'][0]['distance_macro'] for x in arms if x['method']==method]
        table.append(dict(method=method,distance=float(np.mean([r['distance_macro'] for r in rows])),
            distance_seed_range=[float(min(r['distance_macro'] for r in rows)),float(max(r['distance_macro'] for r in rows))],
            coverage_010=float(np.mean([r['coverage_010'] for r in rows])),
            improvement_vs_initial=1-float(np.mean([r['distance_macro'] for r in rows]))/float(np.mean(baseline)),
            joint=float(np.mean([r['joint'] for r in rows])),induced_joint=float(np.mean([r['induced_joint'] for r in rows])),
            late_service=float(np.mean([r['late_service'] for r in rows])),redundant=float(np.mean([r['redundant_005'] for r in rows])),
            unknown=float(np.mean([r['unknown_profiles'] for r in rows]))))
    def source_dist(method):
        result={}
        for x in arms:
            if x['method']!=method:continue
            for c in x['evaluation'][-1]['contexts']:result.setdefault(c['source_id'],[]).append(c['mean_distance'])
        return {k:float(np.mean(v)) for k,v in result.items()}
    ci={}
    for base in ['error_td3','random','sobol']:
        x=source_dist(base);y=source_dist('space_td3');delta=np.array([x[k]-y[k] for k in sorted(x)])
        boots=np.random.default_rng(42).choice(delta,(2000,len(delta)),replace=True).mean(1)
        ci[base]=dict(distance_reduction=float(delta.mean()),ci95=np.quantile(boots,[.025,.975]).tolist(),sources=len(x))
    native_root=project/'artifacts/native_information'/a.native_run
    native=[];checks=0;statuses=Counter()
    for f in sorted(native_root.glob('*/record.json')):
        r=read(f);rows=r['rows'];assert len(rows)==6 and all(x['physical_valid'] for x in rows)
        for row in rows:
            assert row['max_bus_residual_mva']<1e-5
            assert min(row['voltage'])>=.93-1e-6 and max(row['voltage'])<=1.07+1e-6
            assert row['max_line_loading_ratio']<=1.+1e-6
            assert .2-1e-7<=min(row['ess_energy_mwh']) and max(row['ess_energy_mwh'])<=1.8+1e-7
            assert min(row['backup_energy_remaining_mwh'])>=-1e-7
            assert row['battery_balance_error_kwh']<1e-5 and row['ess_balance_error_mwh']<1e-7
            statuses[row['normal_regulation']['status']]+=1;checks+=1
        h=r['hardware'];assert h['line_limit_ka']==.9 and h['voltage_limits_pu']==[.93,1.07]
        native.append(dict(source_id=r['source_id'],action=r['candidate_id'],events=r['event'],known=r['event_known'],
            first_hour_discharge_mwh=float(np.maximum(np.array(rows[0]['ess_energy_mwh'])-.2,0).sum()*.95),
            future_request_L1=float(np.abs([x['candidate_request_mw'] for x in rows[1:]]).sum()),
            Q_states=[x['normal_regulation']['status'] for x in rows]))
    assert len(native)==8
    result=dict(complete=True,query_budget=128,seeds=[42,43,44],contexts=24,sources=6,initial_rows=192,pool_rows=384,
        table=table,source_CI=ci,privileged_greedy_distance=float(np.mean(oracle_dist)),
        native=native,native_verified_hours=checks,native_Q_status=dict(statuses),
        risk_forecaster_training_calls=0,old_rl_training_calls=0,
        acquisition_training_calls_pilot=6,scope='finite-pool V4 acquisition; separate V5 physical sanity')
    (root/'analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='native'},indent=2))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    colors={'random':'#909BA5','sobol':'#37877B','error_td3':'#4477A1','space_td3':'#B45D58'}
    names={'random':'Random','sobol':'Sobol','error_td3':'Error-guided critics','space_td3':'Profile-guided critics'}
    out=project/'docs/information_acquisition_2026-11-02';out.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(13,9.5));fig.subplots_adjust(left=.08,right=.96,top=.88,bottom=.1,hspace=.4,wspace=.3)
    fig.suptitle('Acquiring diverse resource and control-response experience',x=.08,ha='left',fontsize=18,weight='bold')
    fig.text(.08,.92,'Shared finite simulator pool | 192 initial trajectories | 128 added queries | 3 learning seeds',color='#697c88')
    for method in colors:
        runs=[x for x in arms if x['method']==method];budgets=[e['budget'] for e in runs[0]['evaluation']]
        y=np.array([[e['distance_macro'] for e in x['evaluation']] for x in runs])
        axes[0,0].plot(budgets,y.mean(0),label=names[method],color=colors[method],marker='o',ms=4)
        axes[0,0].fill_between(budgets,y.min(0),y.max(0),color=colors[method],alpha=.12)
    axes[0,0].set(title='a  Mechanism coverage improves with queries',xlabel='Added simulator answers',ylabel='Mean distance to observed profiles (lower is better)')
    axes[0,0].legend(frameon=False,fontsize=8)
    xs=np.arange(4)
    axes[0,1].bar(xs,[r['distance'] for r in table],color=[colors[r['method']] for r in table],width=.6)
    axes[0,1].axhline(result['privileged_greedy_distance'],color='#394552',ls='--',label='Privileged greedy diagnostic')
    axes[0,1].set(xticks=xs,xticklabels=['Random','Sobol','Error\ncritics','Profile\ncritics'],ylabel='Mean remaining coverage distance',title='b  Broad coverage is not yet optimal')
    axes[0,1].legend(frameon=False,fontsize=8)
    for i,r in enumerate(table):
        y=[x['evaluation'][-1]['joint'] for x in arms if x['method']==r['method']]
        axes[1,0].scatter([i]*len(y),y,color=colors[r['method']],s=35)
        axes[1,0].plot([i-.15,i+.15],[np.mean(y)]*2,color=colors[r['method']],lw=3)
    axes[1,0].set(xticks=xs,xticklabels=['Random','Sobol','Error\ncritics','Profile\ncritics'],ylabel='Service–RE branches found / 128 queries',title='c  Risk density and coverage are different')
    for sid,marker in [('2024-03-04_s101','o'),('2024-04-01_s102','s')]:
        rr=[r for r in native if r['source_id']==sid];ref=next(r for r in rr if r['action']==0)
        for r in rr:
            xx=r['first_hour_discharge_mwh']-ref['first_hour_discharge_mwh'];yy=r['future_request_L1']-ref['future_request_L1']
            axes[1,1].scatter(xx,yy,marker=marker,s=65,color='#37877B' if r['action']==0 else '#B45D58')
            axes[1,1].annotate(f"{sid[5:10]} a{r['action']}",(xx,yy),xytext=(4,4 if r['action']%2==0 else -13),textcoords='offset points',fontsize=7)
    axes[1,1].axhline(0,color='#ccd5da',lw=1);axes[1,1].axvline(0,color='#ccd5da',lw=1)
    axes[1,1].set(title='d  Separate V5 continuation sanity',xlabel='Change in deliverable ESS energy after hour 1 (MWh)',ylabel='Change in future request activity (L1, MWh)')
    for ax in axes.ravel():ax.grid(axis='y',color='#e8edf0',alpha=.8);ax.set_axisbelow(True)
    fig.text(.08,.035,'Bands in a: range across seeds. Panel d: 8 V5 branches, not a sampler comparison. Profiles describe resource projections, not full feasible volume.',fontsize=8,color='#697c88')
    fig.savefig(out/'comparison.png',dpi=220);fig.savefig(out/'comparison.svg');plt.close(fig)
    svg=out/'comparison.svg';svg.write_text('\n'.join(x.rstrip() for x in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    header='| 方法 | 剩余覆盖距离↓ | 较初始经验下降 | Service–RE发现数 | 延迟Service发现数 |\n|---|---:|---:|---:|---:|\n'
    rows=''.join(f"| {names[r['method']]} | {r['distance']:.5f} | {r['improvement_vs_initial']:.1%} | {r['joint']:.1f}/128 | {r['late_service']:.1f}/128 |\n" for r in table)
    md=f'''# 信息采集首轮对照：资源能力与控制响应画像

本轮只实施采集学习与物理接续核验，风险预警器、专家、融合和旧RL训练均为0。新采集学习器在CPU上训练，pilot为两种目标×3seed共6次，另有smoke两次。原V4/V5代码及硬件保持不变。

![四联结果图](comparison.png)

## 主要结果

{header}{rows}

以上为3seed均值。距离以同一384条历史新增候选的完整机制画像为固定评价全集，初始192条经验另列。原始数据来自6个训练source、12个物理初态和24个偏好情境；它是共享有限缓存上的新增答案揭示实验，不是全新source或连续动作域证明。历史建库为192+384条轨迹，本次每臂另计128次答案访问。

机制画像驱动相对误差驱动的覆盖改进值得保留；但当前没有稳定超过random/Sobol，说明仅将奖励换成已知画像距离还不足以构成高效采集策略。误差驱动更倾向于获得Service–RE正例，而不一定覆盖更多不同响应。两者回答的是不同问题。

## 画像及采集接口

画像由未来各小时ESS可放/可吸能量、下小时可放/可充功率、备用余量，以及后五小时MPC请求和偏好组成，三组按固定物理标尺平衡。风险标签、损失、等待、弃发严重度不参与画像。当前是资源方向投影和控制响应，未计算完整网络—服务可行域。

采集器只获得49维当前可见状态与51维归一化候选动作；选中之后Oracle才揭示未来画像、误差效用和标签。风险侧效用由同一冻结V4初始模型只读前向获得。两学习臂采用相同双critic、预热和更新预算，25%探索，唯一主要变化是奖励。它是TD3式critic引导的有限池上下文采集，连续actor虽有更新但不用于本轮候选选取。

新颖性随已揭示集合变化，replay按当前同情境集合重算并排除自身；未知画像不补零。Sobol将新低差异点映射到同情境最近未查询候选，random同池均匀。所有方法使用共同情境访问顺序。事后贪心诊断使用全库未来画像，剩余距离为{result['privileged_greedy_distance']:.5f}；它有特权信息，仅用于衡量差距，不能作为可部署方法。

## 来源统计

以source为单位，将3seed结果先平均，再对6source配对bootstrap 2000次。下列正差表示机制画像目标剩余距离更小。训练seed变化与source置信区间分开记录。

'''
    for base,v in ci.items():md+=f"相对{names[base]}：距离下降{v['distance_reduction']:.5f}，95% CI [{v['ci95'][0]:.5f}, {v['ci95'][1]:.5f}]。\n\n"
    md+='''## V5真实接续小验证

从两个V4保存的train初态出发，分别执行参考、限充、释放资源和一个Sobol动作，按V5常规有界Q实际演化六小时。8条分支、48小时全部AC核验有效；Voltage已确认正例0、未知0，Service–RE共同事件1条。该小实验用于接续和接口验证，不作为采集器优胜证据，也不代表已解决全量188小时数值未知或175小时反送边界。

每小时核验节点平衡、电压、线路、储能及备用边界、车辆和ESS能量守恒。多初始化仅在原Q求解失败后尝试pf初始化，约束和容差不变。反送许可仍单列，数值失败仍未知。保存每步决策时view及完整物理记录，后续可进一步计算正常服务调度域。

## 下一步

这一轮已把“更多危险后果”与“更丰富机制经验”分开量化。更值得做的定向改进是将当前资源方向投影升级为带期限、网络和正常处置预算的可调能力指标，并采集相近动作在恢复边界两侧的对照；预警器仍可暂缓。避免只最大化损害或画像变化。本文未在看到结果后改阈值、删seed或重选运行。

## 执行命令

```powershell
Set-Location -LiteralPath 'D:\\博士资料包\\科研\\科研\\AAAI_RL\\tarp_37_information_acquisition'
powershell -NoProfile -ExecutionPolicy Bypass -File .\\scripts\\run_information.ps1 -Mode pilot -RunId information_pilot_v1 -Resume
```

已有运行的Resume核验并复用结果；新实验使用新RunId。V5原始记录位于artifacts/native_information/v5_information_sanity_v1，采集回放位于artifacts/information/information_pilot_v1。完整画像、查询顺序及来源统计均保留。
'''
    (out/'results.md').write_text(md,encoding='utf-8')
    import markdown
    html=markdown.markdown(md,extensions=['tables','fenced_code'])
    style='body{max-width:1160px;margin:35px auto;padding:0 28px;font:17px/1.8 "Microsoft YaHei",sans-serif;color:#263b49}h1{font-size:29px}h2{font-size:23px;margin-top:36px}img{width:100%}table{border-collapse:collapse;width:100%;font-size:15px}th,td{border:1px solid #d9e2e7;padding:9px}th{background:#edf3f6}pre{padding:15px;background:#f3f6f8;overflow:auto}code{font-size:14px}'
    (out/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>信息采集对照</title><style>'+style+'</style><body>'+html+'</body></html>',encoding='utf-8')
    print(out/'index.html')


if __name__=='__main__':main()
