"""Compact physical closure report, rebuilt only from published records."""
import argparse
from pathlib import Path
import html
import json
import numpy as np
from .assets import PROJECT,sha
from .collection import read,write,verify


def build(run_id):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=PROJECT/'artifacts/runs'/run_id
    records=[]
    for p in sorted((root/'sources').glob('*/complete.json')):
        verify(read(p)); records+=read(p.parent/'records.json')
    summary=read(root/'analysis/summary.json'); pairs=read(root/'analysis/paired_changes.json')
    # Keep engineering plots readable for larger collections. Tables and
    # summaries still use all records; the graphic is explicitly illustrative.
    records=records[:12]
    out=root/'analysis'; names=['Voltage','Network capacity','Charging service','RE curtailment']
    labels=[f"{r['family']} / p{r['profile_id']}" for r in records]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,2,figsize=(13.5,8.3),layout='constrained')
    first=np.array([[sum(r['rows'][0]['actual_device_mw'][:37]),sum(r['rows'][0]['actual_device_mw'][37:47]),sum(r['rows'][0]['actual_device_mw'][47:])] for r in records])
    a=axs[0,0]; lim=max(float(np.max(np.abs(first))),.1)
    im=a.imshow(first,cmap='BrBG',vmin=-lim,vmax=lim,aspect='auto'); a.set_xticks(range(3),['SCS','FCS','ESS']); a.set_yticks(range(len(labels)),labels)
    for i in range(len(first)):
        for j in range(3): a.text(j,i,f'{first[i,j]:+.2f}',ha='center',va='center',color='black',fontsize=8)
    a.set_title('a  First-hour realized delivery (MW)',loc='left',weight='bold'); fig.colorbar(im,ax=a,shrink=.8)
    b=axs[0,1]
    for r in records:
        if r['candidate_id']==0: continue
        ref=next(x for x in records if x['source_id']==r['source_id'] and x['pool_hour']==r['pool_hour'] and x['profile_id']==r['profile_id'] and x['candidate_id']==0)
        h=min(len(r['rows']),len(ref['rows']))
        delta=np.abs(np.array([x['actual_device_mw'] for x in r['rows'][:h]])-np.array([x['actual_device_mw'] for x in ref['rows'][:h]])).sum(1)
        b.plot(np.arange(1,h+1),delta,'o-',label=f"{r['family']} / p{r['profile_id']}",lw=1.6,ms=4)
    b.axvline(1.5,color='#9AA6AF',ls=':',lw=1); b.set(xlabel='Hour after intervention',ylabel='L1 delivered-power change vs reference (MW)')
    b.set_title('b  Initial action changes later MPC responses',loc='left',weight='bold'); b.legend(frameon=False,fontsize=7)
    c=axs[1,0]; events=np.array([r['event'] for r in records]); known=np.array([r['event_known'] for r in records])
    c.imshow(np.ma.array(events,mask=~known),cmap=matplotlib.colors.ListedColormap(['#E9EFF2','#B65D4E']),vmin=0,vmax=1,aspect='auto')
    c.set_yticks(range(len(labels)),labels); c.set_xticks(range(4),names,rotation=15,ha='right')
    for i in range(len(events)):
        for j in range(4): c.text(j,i,'?' if not known[i,j] else 'Event' if events[i,j] else 'No event',ha='center',va='center',fontsize=7)
    c.set_title('c  Recomputed six-hour physical event labels',loc='left',weight='bold')
    d=axs[1,1]
    for i,r in enumerate(records):
        rows=[x for x in r['rows'] if x['physical_valid']]
        if not rows: continue
        d.plot(range(1,len(rows)+1),[x['max_line_loading_ratio']*100 for x in rows],'-o',ms=3,lw=1.2,label=labels[i])
    d.axhline(100,color='#B65D4E',ls='--',lw=1,label='0.9 kA rating')
    d.set(xlabel='Hour after intervention',ylabel='Actual maximum line loading (%)',ylim=(0,max(110,summary.get('max_line_loading_ratio',0)*110)))
    d.set_title('d  Actual post-recourse AC current remains bounded',loc='left',weight='bold'); d.legend(frameon=False,fontsize=6,ncol=2)
    fig.suptitle('37nodes: intervention, resource evolution and verified AC execution',fontsize=15,weight='bold')
    fig.savefig(out/'physical_closure.png',dpi=200); fig.savefig(out/'physical_closure.svg'); plt.close(fig)
    def f(x): return '—' if x is None else f'{x:.6g}' if isinstance(x,(int,float)) else str(x)
    metrics=[('独立运行日期',summary['sources']),('分支 / 未来小时',f"{summary['branches']} / {summary['future_hours']}"),('AC有效 / 未知小时',f"{summary['physical_valid_hours']} / {summary['unknown_hours']}"),
        ('实际电压范围 pu',f"{f(summary['min_voltage_pu'])} – {f(summary['max_voltage_pu'])}"),('实际最大线路负载率 %',summary['max_line_loading_ratio']*100 if summary['max_line_loading_ratio'] is not None else None),
        ('最大AC节点不平衡 MVA',summary['max_ac_residual_mva']),('最大车辆电量记账误差 kWh',summary['max_battery_balance_error_kwh']),('未来偏好不应改变首小时的核验次数',summary['first_hour_preference_invariance_checks'])]
    table='<table><tr><th>核验项</th><th>结果</th></tr>'+''.join('<tr><td>'+html.escape(k)+'</td><td>'+html.escape(f(v))+'</td></tr>' for k,v in metrics)+'</table>'
    pairrows=''.join('<tr><td>'+html.escape(p['family'])+'</td><td>'+str(p['profile_id'])+'</td><td>'+f(p['future_control_l1_mwh'])+'</td><td>'+f(p['differences']['total_operating_cost'])+'</td><td>'+f(p['differences']['queue_vehicle_min'])+'</td><td>'+f(sum(p['terminal_ess_delta_mwh']))+'</td></tr>' for p in pairs)
    paired='<table><tr><th>首小时干预</th><th>未来偏好</th><th>后五小时动作L1变化 MWh</th><th>费用变化 native units</th><th>排队变化 车·分钟</th><th>末端ESS变化 MWh</th></tr>'+pairrows+'</table>'
    text=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>37nodes统一物理闭环核验</title>
<style>body{{font:17px/1.8 "Microsoft YaHei",sans-serif;color:#243741;max-width:1260px;margin:38px auto;padding:0 24px}}h1{{font-size:29px}}h2{{font-size:22px;margin-top:34px}}table{{border-collapse:collapse;width:100%;font-size:15px}}th,td{{padding:9px 12px;border-bottom:1px solid #dce5e8;text-align:left}}th{{background:#edf3f5}}img{{width:100%}}.note{{padding:16px;background:#f3f7f8;border-left:4px solid #287D88}}code{{overflow-wrap:anywhere}}</style>
<h1>37nodes统一物理闭环核验</h1><p>在原始道路、电网、线路阻抗、机组能力和充电站配置上，以0.9 kA、0.93–1.07 pu作为统一运行合同。原生车辆由MPC暖启动，首小时施加候选动作，此后继续有限信息动态偏好MPC；真实交通按15分钟记录，配电网按小时结算。</p>
<p class="note">本页用于检查干预—资源—后续调度链条与物理执行是否闭合。当前为{summary['sources']}个运行日期。电气事件表示完整常规P/Q、风光利用、备用和保供计划在真实交付下的AC约束压力；额外纠正后的实际电压、电流和机组约束另行验证。电压偏差和节点压差不单独计为事件，反送许可偏离与热容量超限分列。</p>
<h2>小规模运行结果</h2>{table}<img src="physical_closure.png" alt="Physical closure panel">
<p>组合图展示按source排序的前{len(records)}条分支，汇总和配对表使用全部数据。红色为六小时窗口中的已知事件；灰色为完整观测后的未发生；问号为证据不足。风、光分别按窗口弃发比例判断。30分钟等待和10%弃发均为明确的场景标准，后者不解释为年度政策合规结论。</p>
<h2>同状态干预如何改变后续调度</h2>{paired}
<p>上述差异均相对相同状态、外部轨迹和未来偏好下的MPC参考行动。后五小时L1只统计实际交付功率，末端ESS单独记录，避免把透支库存误判成纯收益。费用保留原始机组曲线与场景恢复代价，AC损耗正则单列，不冒称实测货币费用。</p>
<h2>数据与模型边界</h2><p>物理版本：<code>{summary['physics_version']}</code>；风险版本：<code>{summary['risk_contract']}</code>。原RL与旧专家训练调用均为0。当前输出是新环境数据和学习接口的核验结果，正式新预警器、采样RL和专家融合需按重训计划执行。</p>
<p><a href="summary.json">物理汇总</a> · <a href="paired_changes.json">配对差异</a> · <a href="physical_closure.svg">SVG组合图</a></p></html>'''
    (out/'index.html').write_text(text,encoding='utf-8')
    inputs={str(root/'analysis/summary.json'):sha(root/'analysis/summary.json'),str(root/'analysis/paired_changes.json'):sha(root/'analysis/paired_changes.json'),str(Path(__file__)):sha(__file__)}
    write(out/'report_manifest.json',dict(inputs=inputs,outputs={str(out/p):sha(out/p) for p in ['index.html','physical_closure.png','physical_closure.svg']}))
    print(out/'index.html')


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--data-run',required=True); build(parser.parse_args().data_run)
