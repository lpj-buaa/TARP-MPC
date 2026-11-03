from pathlib import Path
from collections import defaultdict,Counter
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import markdown

NAMES={'ppo':'Sequential PPO','myopic':'Myopic PPO','group':'Group-relative','greedy':'Model greedy','local':'Local interpolation','bracket_random':'Bracket + Random','sobol':'Sobol','random':'Random'}
COLORS={'ppo':'#147D92','myopic':'#7B8690','group':'#A37BA8','greedy':'#799A71','local':'#C26A52','bracket_random':'#344A70','sobol':'#B5A052','random':'#A6A09B'}


def read(p):return json.loads(Path(p).read_text('utf-8'))
def macro(rows,metric):
    groups=defaultdict(list)
    for r in rows:
        if r.get(metric) is not None:groups[r['source_id']].append(r[metric])
    values={s:float(np.mean(v)) for s,v in groups.items()}
    return dict(mean=float(np.mean(list(values.values()))) if values else None,sources=values)
def table(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)
def number(v):return '无支持' if v is None else f'{v:.4f}'


def report(root):
    root=Path(root);out=root/'report';out.mkdir(exist_ok=True)
    data=read(root/'dataset.json');diagnostics=read(root/'diagnostics.json');evaluation=read(root/'evaluation.json');cases=data['cases']
    own=[r for r in evaluation['rows'] if r['role']=='development_test'];methods=list(NAMES)
    summary={str(b):{m:{metric:macro([r for r in own if r['method']==m and r['budget']==b],metric) for metric in ['utility','distance','boundary_recall','unknown_queries']} for m in methods} for b in [8,16,24]}
    fits=[read(p) for p in sorted(root.glob('response_s*/fit.json'))]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(16,9),layout='constrained')
    for j,key in enumerate(['selection_loss','mae','delta_mae']):
        for fit in fits:
            h=fit['history'];x=[r['epoch'] for r in h]
            axes[0,j].plot(x,[r['train'][key] for r in h],color='#147D92',alpha=.55,label='Train' if fit==fits[0] else None)
            axes[0,j].plot(x,[r['validation'][key] for r in h],'--',color='#C26A52',alpha=.7,label='Tune' if fit==fits[0] else None)
        axes[0,j].set(title=['a  Comparable training / tuning loss','b  Masked outcome MAE','c  Paired action-effect MAE'][j],xlabel='Epoch',ylabel=key);axes[0,j].legend(frameon=False)
    groups=['bounds','reserve_effort','gap','followup','preference'];rs=diagnostics['response_rows']
    matrix=[]
    for role in ['train','tune','development_test']:
        row=[]
        members={'train','meta_train'} if role=='train' else {'tune','meta_tune'} if role=='tune' else {role}
        for group in groups:
            values=[dict(source_id=r['source_id'],value=r['groups'][group]['mae']) for r in rs if r['role'] in members]
            row.append(macro(values,'value')['mean'])
        matrix.append(row)
    image=axes[1,0].imshow(np.array(matrix,float),cmap='YlGnBu',aspect='auto')
    for i in range(3):
        for j in range(5):axes[1,0].text(j,i,number(matrix[i][j]),ha='center',va='center',fontsize=9)
    axes[1,0].set(xticks=range(5),xticklabels=['Bounds','Reserve','Gap','Dispatch','Preference'],yticks=range(3),yticklabels=['Train','Tune','Dev holdout'],title='d  Consequence groups: source-macro MAE');fig.colorbar(image,ax=axes[1,0],shrink=.7)
    for method,color in [('model','#147D92'),('local','#C26A52')]:
        values=[]
        for h in [0,4,8]:values.append(macro([r for r in diagnostics['ranking_rows'] if r['role']=='development_test' and r['method']==method and r['history_extra']==h],'spearman')['mean'])
        axes[1,1].plot([0,4,8],values,'-o',color=color,label=method)
    axes[1,1].set(title='e  Immediate information-value ranking',xlabel='Shared additional Random observations',ylabel='Source-macro Spearman');axes[1,1].legend(frameon=False)
    for m in methods:axes[1,2].plot([8,16,24],[summary[str(b)][m]['utility']['mean'] for b in [8,16,24]],'-o',label=NAMES[m],color=COLORS[m])
    axes[1,2].set(title='f  Matched-budget acquisition on dev holdouts',xlabel='Paid branch queries',ylabel='Terminal utility');axes[1,2].legend(frameon=False,fontsize=8,ncol=2)
    for ax in axes.ravel():ax.grid(alpha=.13)
    fig.suptitle('Interim learning diagnosis — original formal collection paused',fontsize=16,fontweight='bold')
    fig.savefig(out/'learning_diagnostic.png',dpi=200);fig.savefig(out/'learning_diagnostic.svg');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for j,source in enumerate(sorted({r['source_id'] for r in own})):
        for m in methods:
            values=[macro([r for r in own if r['source_id']==source and r['method']==m and r['budget']==b],'utility')['mean'] for b in [8,16,24]]
            axes[j].plot([8,16,24],values,'-o',color=COLORS[m],label=NAMES[m])
        axes[j].set(title=source,xlabel='Paid branch queries',ylabel='Terminal utility');axes[j].grid(alpha=.15)
    axes[-1].legend(fontsize=8,ncol=2,frameon=False);fig.savefig(out/'holdout_sources.png',dpi=200);fig.savefig(out/'holdout_sources.svg');plt.close(fig)
    support={r:dict(sources=len({c['source_id'] for c in cases if c['role']==r}),contexts=sum(c['role']==r for c in cases),branches=sum(len(c['ids']) for c in cases if c['role']==r),verified=sum(sum(c['valid']) for c in cases if c['role']==r)) for r in sorted({c['role'] for c in cases})}
    metrics=diagnostics['response_summary'];train_rows=[]
    for f in fits:train_rows.append([f['seed'],f['epochs'],f['best_epoch'],number(f['best_train']['mae']),number(f['best_validation']['mae']),f['optimizer_steps'],f"{f['elapsed_seconds']:.1f}",f"{f['peak_cuda_allocated_bytes']/2**20:.1f}"])
    response_rows=[[role,*[number(metrics[role][m]['mean']) for m in ['mae','zero_mae','delta_mae','delta_zero_mae','direction_accuracy']]] for role in metrics]
    rows=[[NAMES[m],*[number(summary[str(b)][m]['utility']['mean']) for b in [8,16,24]],number(summary['16'][m]['boundary_recall']['mean']),number(summary['16'][m]['unknown_queries']['mean'])] for m in methods]
    ranking_rows=[]
    for method in ['model','local']:
        selected=[r for r in diagnostics['ranking_rows'] if r['role']=='development_test' and r['method']==method]
        ranking_rows.append([method,number(macro(selected,'spearman')['mean']),sum(r['spearman'] is not None for r in selected),len(selected),number(macro(selected,'regret')['mean'])])
    primary=summary['16'];delta={m:primary['ppo']['utility']['mean']-primary[m]['utility']['mean'] for m in methods if m!='ppo'}
    analysis=dict(development_only=True,original_confirm_used=0,support=support,summary=summary,ppo_minus_comparator_budget16=delta,
                  response_summary=metrics,ranking_summary=diagnostics['ranking_summary'])
    (root/'analysis.json').write_text(json.dumps(analysis,ensure_ascii=False,indent=2,allow_nan=False),'utf-8')
    text=f'''# 暂停采集后的现有样本诊断

原正式采集已按用户要求暂停，7个进程保留。此次只读76个完整情境、2432条分支；原最终confirm未参与。两个开发留出source为2024-04-01_s102与2024-08-19_s112，来自原训练角色；10个source训练、4个source调参、3个source作cal支持检查。该报告回答现有样本能否学会后果和采集价值，不代替最终正式比较。

## 数据支持

{table(['角色','source','情境','分支','完整后果画像'],[[r,v['sources'],v['contexts'],v['branches'],v['verified']] for r,v in support.items()])}

未知标签保留mask，只有完整核验画像进入终局效用分母；不完整的原confirm池没有被截成更容易的候选池。

## 训练与验证是否匹配

三个seed均使用原136639参数后果模型，监督、学习率、batch和早停保持不变。逐epoch增加无梯度、无随机采样监测；跨训练与调参比较相同无权重口径，原正例加权优化损失单独记录。所有模型仅按原tune/meta_tune角色选择。

{table(['seed','训练轮数','best epoch','训练MAE','调参MAE','梯度步数','训练秒','峰值PyTorch显存MiB'],train_rows)}

![学习、后果与采集诊断](learning_diagnostic.png)

{table(['角色','后果MAE','零变化MAE','行动增量MAE','零增量MAE','显著变化方向正确率'],response_rows)}

行动增量比较同情境候选与参考，双方标签已知才计入。方向指标只统计归一化真实增量绝对值大于0.01的单元，此阈值用于区分有效变化，不是工程报警标准。图a至c比较训练/调参可学习性；图d检查哪类物理后果主导误差。

## 信息价值排序与预算化采集

各模型使用完全相同的公共初始3条和预定Random增量历史，比较预测即时边际效用与已知真实边际效用。原始真值平坦时不报告虚假的Spearman；该排序评价和多步终局规划分别报告。

{table(['排序器','Spearman','有区分度的排序组','排序组总数','Top1边际遗憾'],ranking_rows)}

{table(['采集方法','预算8效用','预算16效用','预算24效用','预算16边界率','预算16未知查询'],rows)}

预算16下，序贯PPO相对即时PPO为{delta['myopic']:+.4f}，相对局部插值为{delta['local']:+.4f}，相对括区＋Random为{delta['bracket_random']:+.4f}，相对Random为{delta['random']:+.4f}。各seed、两个开发source及四种时间/偏好情境在相同分母上比较；逐源结果如下。

![两个开发source的预算曲线](holdout_sources.png)

## 诊断解释与后续容量对照

训练和调参均存在较高误差时，优先核查容量、优化和输入表达；训练误差小而留出误差大时，优先核查跨source经验；后果预测可用而边际价值排序差时，核查后果到采集目标的映射；即时排序可用而PPO不胜强简单方法时，再分析多步规划的实际空间。

本轮使用当前轻量架构建立监测基线，不提前更换正式架构。用户恢复并完成当前正式实验后，容量对照固定数据hash、source、输入、监督和查询预算，先比较小/宽MLP，再对参数量接近的宽MLP与空间时序模型进行比较。结构模型若增加已有历史输入，MLP也应获得同样信息，分别识别容量、结构和信息收益。实际梯度步数、训练时间、显存和独立source表现同时记录。

原采集仍暂停。暂停保留内存态；恢复前须核对pause_state.json中的PID与创建时间，不可重复启动或重复恢复。冻结MPC、容量、阈值、旧RL和预警器均未改动。
'''
    (out/'report.md').write_text(text,'utf-8');html=markdown.markdown(text,extensions=['tables','fenced_code'])
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>中期学习诊断</title><style>body{max-width:1300px;margin:30px auto;padding:0 28px;font:17px/1.8 "Microsoft YaHei",sans-serif;color:#263d49}img{width:100%}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;border-bottom:1px solid #dce4e8}th{background:#edf3f5}code{font-size:13px}</style>'+html,'utf-8')
    print(str(out/'index.html'));return analysis


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();report(a.root)
