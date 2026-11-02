from pathlib import Path
from collections import defaultdict,Counter
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import markdown

NAMES={'ppo':'Sequential PPO','myopic':'Myopic PPO','group':'Group-relative sequence','random':'Random','sobol':'Parameter Sobol',
       'local':'Local interpolation','bracket_random':'Bracket + Random','greedy':'Model-based greedy'}
CN={'ppo':'序贯PPO','myopic':'即时PPO','group':'组相对序列策略','random':'Random','sobol':'参数Sobol','local':'局部插值','bracket_random':'括区＋Random','greedy':'后果模型即时贪心'}
COLORS={'ppo':'#087F8C','myopic':'#7E8792','group':'#9F6B99','random':'#A89C87','sobol':'#C5AA53','local':'#BB5A4C','bracket_random':'#313D64','greedy':'#5E9B62'}


def read(p):return json.loads(Path(p).read_text('utf-8'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),'utf-8')


def source_values(rows,method,budget,metric):
    groups=defaultdict(list)
    for r in rows:
        if r['method']==method and r['budget']==budget and r[metric] is not None:groups[r['source_id']].append(r[metric])
    return {k:float(np.mean(v)) for k,v in groups.items()}


def contrast(rows,a,b,budget,metric):
    left=source_values(rows,a,budget,metric);right=source_values(rows,b,budget,metric);keys=sorted(left.keys()&right.keys())
    if not keys:return dict(difference=None,ci95=None,n_sources=0)
    d=np.array([left[k]-right[k] for k in keys]);rng=np.random.default_rng(20261102)
    boots=d[rng.integers(len(d),size=(2000,len(d)))].mean(1)
    return dict(difference=float(d.mean()),ci95=np.quantile(boots,[.025,.975]).tolist(),n_sources=len(keys),sources=dict(zip(keys,d.tolist())))


def table(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root);out=root/'report';out.mkdir(exist_ok=True)
    data=read(root/'dataset.json');evaluation=read(root/'evaluation.json')
    scope=data.get('scope','');protocol=read(root/'learning_protocol.json')
    pilot='pilot' in scope or protocol['mode']=='smoke'
    rows=[r for r in evaluation['rows'] if r['role']=='confirm'];primary_role='confirm'
    if not rows:
        if pilot:rows=evaluation['rows'];primary_role='engineering_resubstitution'
        else:
            write(root/'analysis.json',dict(supported=False,primary_role='confirm_unsupported',reason='No confirm contexts with full verified response descriptors',
                                            available_roles=sorted({r['role'] for r in evaluation['rows']})))
            (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Confirm数据不支持正式比较</h1><p>没有具备完整核验后果画像的confirm情境；未用tune或cal数据替代。请查看response_diagnostics.json中的分项误差和支持情况。</p>','utf-8')
            print(str(out/'index.html'));return
    methods=list(NAMES);summary={}
    for budget in [8,16,24]:
        summary[str(budget)]={}
        for method in methods:
            summary[str(budget)][method]={}
            for metric in ['utility','distance','boundary_recall','unknown_queries']:
                values=source_values(rows,method,budget,metric)
                summary[str(budget)][method][metric]=dict(mean=float(np.mean(list(values.values()))) if values else None,n_sources=len(values),sources=values)
    comparisons={m:{metric:contrast(rows,'ppo',m,16,metric) for metric in ['utility','distance','boundary_recall']} for m in methods if m!='ppo'}
    write(root/'analysis.json',dict(supported=True,summary=summary,comparisons=comparisons,primary_role=primary_role,source_bootstrap=2000,development_pilot=pilot))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    for method in methods:
        axes[0,0].plot([8,16,24],[summary[str(b)][method]['utility']['mean'] for b in [8,16,24]],'-o',color=COLORS[method],label=NAMES[method],lw=2 if method=='ppo' else 1.2)
        axes[0,1].plot([8,16,24],[summary[str(b)][method]['distance']['mean'] for b in [8,16,24]],'-o',color=COLORS[method],label=NAMES[method],lw=2 if method=='ppo' else 1.2)
    axes[0,0].set(title='a  Verified end-of-budget acquisition utility',xlabel='Paid branch queries per context',ylabel='Source-macro utility');axes[0,0].legend(fontsize=8,ncol=2,frameon=False)
    axes[0,1].set(title='b  Fixed-pool consequence coverage gap',xlabel='Paid branch queries per context',ylabel='Nearest-descriptor distance (lower is better)')
    for i,method in enumerate([m for m in methods if m!='ppo']):
        c=comparisons[method]['utility'];v=c['difference'];lo,hi=c['ci95'];axes[1,0].errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',color=COLORS[method],capsize=3)
    axes[1,0].axvline(0,color='gray',lw=1);axes[1,0].set(yticks=range(7),yticklabels=[NAMES[m] for m in methods if m!='ppo'],xlabel='Sequential PPO minus comparator; source 95% CI',title='c  Paired differences at budget16');axes[1,0].invert_yaxis()
    for method in ['ppo','myopic','group']:
        records=[read(p)['history'] for p in (root/'policies').glob(method+'_s*/fit.json')]
        if records:
            batches=[r['batch'] for r in records[0]];values=np.array([[r['tune_utility'] for r in h] for h in records])
            axes[1,1].plot(batches,values.mean(0),'-o',color=COLORS[method],label=NAMES[method]);axes[1,1].fill_between(batches,values.min(0),values.max(0),color=COLORS[method],alpha=.1)
    axes[1,1].set(title='d  Learning on training sources; selection on tuning',xlabel='Training batch',ylabel='Tuning utility at budget16');axes[1,1].legend(fontsize=8,frameon=False)
    for ax in axes.ravel():ax.grid(alpha=.15)
    fig.suptitle('Sequential experiment design under a frozen MPC controller'+(' — development cache pilot' if pilot else ''),fontsize=15,fontweight='bold')
    fig.savefig(out/'sequence_results.png',dpi=200,bbox_inches='tight');fig.savefig(out/'sequence_results.svg',bbox_inches='tight');plt.close(fig)
    table_rows=[]
    for m in methods:
        s=summary['16'][m];br=s['boundary_recall']['mean']
        table_rows.append([CN[m],f"{s['utility']['mean']:.4f}",f"{s['distance']['mean']:.6f}",f'{br:.1%}' if br is not None else '无支持',s['utility']['n_sources']])
    comparison_rows=[[CN[m],f"{c['utility']['difference']:+.4f}",str([round(v,4) for v in c['utility']['ci95']])] for m,c in comparisons.items()]
    physical=read(root/'collection_complete.json') if (root/'collection_complete.json').exists() else dict(native_simulations=0,mpc_solves=0,branches=data['counts']['branches'])
    diagnostics=read(root/'response_diagnostics.json') if (root/'response_diagnostics.json').exists() else {}
    component_rows=[]
    for group,values in diagnostics.get('response_summary',{}).get('confirm',{}).items():component_rows.append([group,f"{values['mae']:.6f}",f"{values['zero_mae']:.6f}"])
    support=diagnostics.get('calibration_support',{})
    recovery_rows=[]
    for hour in [1,3,5]:
        own=[r for r in diagnostics.get('recovery_rows',[]) if r['role']=='confirm' and r['hour']==hour]
        if not own:continue
        firstseed=min(r['seed'] for r in own);count=[r for r in own if r['seed']==firstseed]
        values=[]
        for metric in ['recall','precision','fpr','brier','ece']:
            by=defaultdict(list)
            for r in own:
                if r[metric] is not None:by[r['source_id']].append(r[metric])
            values.append(f"{np.mean([np.mean(v) for v in by.values()]):.4f}" if by else '无支持')
        recovery_rows.append([f'+{hour}h',sum(r['positive'] for r in count),sum(r['negative'] for r in count),*values])
    delta=comparisons['myopic']['utility'];improved=delta['ci95'][0]>0 and not pilot
    text=f'''# 后果模型辅助的序贯信息采集

{'本报告是六个已研究历史source上的开发缓存验证；不作为新confirm结论。' if pilot else '本报告评价扩展原生仿真银行中冻结后的采集策略；训练、调参与confirm按source隔离。'}

采集一步是查询一条干预＋6小时MPC接续分支。状态包含当前情境、已查动作及真实后果集合、未查候选和绝对剩余预算。每步奖励为固定终局效用的增量，序贯PPO使用gamma1，即时PPO使用gamma0；两者输入、架构和训练episode数相同。组相对策略比较相同情境和预算下的整段查询序列。后果模型在每个采集episode内冻结。

## 同预算主比较：每情境新增16次查询

{table(['方法','终局采集效用↑','固定池画像距离↓','恢复边界定位率','有效source数'],table_rows)}

序贯PPO相对即时PPO的效用变化为{delta['difference']:+.4f}，source配对区间{delta['ci95']}。{'本轮支持序贯学习带来正向增益。' if improved else '应结合配对区间判断序贯机制是否兑现；实现了多步回传并不自动证明采集收益提升。'}

![预算与归因组合图](sequence_results.png)

{table(['序贯PPO减去对照','16预算效用变化','95% source区间'],comparison_rows)}

## 评价口径

效用由固定候选池画像覆盖改善和已核验相邻恢复边界定位组成；存在边界时两项等权，无边界时只评价覆盖改善。初始共同3条分支不计新增查询；每条已选择分支只揭示一次答案。未知查询消耗预算，未知后果不构造假安全/假风险标签。训练奖励可由训练银行内部真值计算，策略输入只能读取已查真值和冻结模型预测。confirm上不更新任何参数。

固定池距离对全部非初始候选平均，已经查询的有效候选距离为0。统计先聚合同source的时间点/偏好及种子，再对source配对bootstrap2000次；没有恢复边界的情境不进入边界率分母。{'开发缓存的24预算已耗尽24个候选，该点是穷尽结果。' if pilot and 'pilot' in scope else '扩展银行每情境29个非初始候选，24预算仍未穷尽候选池。'}

## 后果模型与冻结支持检查

{table(['连续后果组','Confirm模型MAE','零变化MAE'],component_rows) if component_rows else '当前无完整confirm分项误差；未使用其他角色替代。'}

{table(['恢复失败时点','已知正例','已知负例','召回','精确率','误报率','Brier','ECE'],recovery_rows) if recovery_rows else ''}

Cal支持情况：`{support}`。逐时恢复分类的已知正负数、召回、精确率、误报率、Brier和ECE全部保存在response_diagnostics.json；cal与confirm均未用于更新或选择模型。该检查用于区分后果学习与查询规划的能力差异。

## 数据与执行事实

数据说明：{scope}。物理阶段记录：`{physical}`。学习阶段不训练旧RL、旧专家或预警器，也不改变容量、MPC目标或物理阈值。完整行级结果在evaluation.json，训练source清单、查询次数和checkpoint选择记录在各fit.json；每条查询序列均可复核。

本轮提出的机制是将稳定后果学习与预算化查询规划分开。若序贯策略未超过局部插值或括区＋Random，结果将说明哪些部分仍应由明确物理计算承担，不能把全部收益归于RL。
'''
    (out/'report.md').write_text(text,'utf-8');html=markdown.markdown(text,extensions=['tables','fenced_code'])
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>序贯采集结果</title><style>body{max-width:1200px;margin:35px auto;padding:0 25px;font:17px/1.85 "Microsoft YaHei",sans-serif;color:#273b49}img{width:100%}table{border-collapse:collapse;width:100%;font-size:15px}td,th{padding:9px;border-bottom:1px solid #d8e1e5}th{background:#edf3f5}code{font-size:13px}</style>'+html,'utf-8')
    print(str(out/'index.html'))


if __name__=='__main__':main()
