"""Render source-level cache experiments; no fitting, simulation or threshold changes."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import markdown

NAMES={'old_summary':'Old summary','quantitative':'Quantitative input','stable':'Stable supervision',
       'combined':'Combined','local_interpolation':'Local interpolation','random':'Random',
       'bracket_random':'Bracket + Random','parameter_sobol':'Parameter Sobol'}
CN={'old_summary':'旧摘要','quantitative':'空间＋定量经验','stable':'稳定后果监督','combined':'两者结合',
    'local_interpolation':'局部物理插值','random':'Random','bracket_random':'括区＋Random','parameter_sobol':'参数空间Sobol'}
COLORS={'old_summary':'#6B7785','quantitative':'#347AA8','stable':'#A8793A','combined':'#007E78',
        'local_interpolation':'#B45358','random':'#B0A3BC','bracket_random':'#323D62','parameter_sobol':'#B4A55E'}
METHODS=['old_summary','quantitative','stable','combined','local_interpolation']


def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)


def draw(analysis,out,diagnostics):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.labelcolor':'#263747','text.color':'#263747','axes.titleweight':'bold','axes.titlesize':10,'savefig.facecolor':'white'})
    fig,axes=plt.subplots(2,2,figsize=(13.5,9),layout='constrained')
    for ax,metric,title in [(axes[0,0],'spearman','a  Ranking unseen actions\nunder identical histories'),
                             (axes[0,1],'top3_capture','b  True utility captured\nby the predicted top three')]:
        for method in METHODS:
            budgets=[b for b in ([16,32,64] if metric=='top3_capture' else [16,32,64,128]) if f'{b}/{metric}' in analysis['ranking']]
            vals=[analysis['ranking'][f'{b}/{metric}'][method] for b in budgets]
            ax.plot(budgets,[v['mean'] for v in vals],marker='o',lw=2,color=COLORS[method],label=NAMES[method])
            ax.fill_between(budgets,[min(v['seeds'].values()) for v in vals],[max(v['seeds'].values()) for v in vals],
                            color=COLORS[method],alpha=.07)
        ax.axvline(64,color='#74838F',ls=':',lw=1);ax.set(title=title,xlabel='Common paid-query budget',ylabel='Within-source Spearman' if metric=='spearman' else 'Top-three utility / oracle top-three utility')
        ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=8,frameon=False,ncol=2,loc='upper left')
    sources=sorted(analysis['ranking']['64/spearman']['combined']['sources'])
    mat=np.array([[analysis['ranking']['64/spearman'][m]['sources'].get(s,np.nan) for s in sources] for m in METHODS])
    im=axes[1,0].imshow(mat,cmap='RdBu',vmin=-1,vmax=1,aspect='auto')
    axes[1,0].set(xticks=range(len(sources)),xticklabels=[s[:7] for s in sources],yticks=range(len(METHODS)),
                   yticklabels=[NAMES[m] for m in METHODS],title='c  Source-specific ranking\nat the primary budget (64)')
    for i in range(len(METHODS)):
        for j in range(len(sources)):axes[1,0].text(j,i,f'{mat[i,j]:.2f}',ha='center',va='center',color='white' if abs(mat[i,j])>.65 else '#263747')
    fig.colorbar(im,ax=axes[1,0],label='Mean of three training seeds',shrink=.8)
    comps=analysis['comparisons']['ranking/64/spearman'][:5]
    for i,c in enumerate(comps):
        mean=c['difference'];lo,hi=c['ci95'];axes[1,1].errorbar(mean,i,xerr=[[mean-lo],[hi-mean]],fmt='o',color=COLORS[c['left']],capsize=3)
    axes[1,1].axvline(0,color='#74838F',lw=1)
    axes[1,1].set(yticks=range(len(comps)),yticklabels=[NAMES[c['left']]+' − '+NAMES[c['right']] for c in comps],
                  xlabel='Paired Spearman change; source bootstrap 95% CI',title='d  Contributions of input\nand stationary supervision')
    axes[1,1].invert_yaxis();axes[1,1].grid(axis='x',alpha=.15)
    fig.suptitle('Information-value ranking: common evidence, common unseen actions',fontsize=17,fontweight='bold')
    for ext in ['png','svg']:fig.savefig(out/f'figure1_matched_ranking.{ext}',dpi=210,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13.5,9),layout='constrained')
    active_methods=METHODS+['bracket_random','random','parameter_sobol']
    for ax,metric,title in [(axes[0,0],'boundary_recall','a  Exact adjacent recovery\nboundary discovery'),
                            (axes[0,1],'distance','b  Fixed-pool consequence\ncoverage gap')]:
        budgets=[0,8,16,32,64,128]
        for m in active_methods:
            y=[analysis['active'][f'{b}/{metric}'][m]['mean'] for b in budgets]
            ax.plot(budgets,y,marker='o',ms=3,lw=2 if m in ['combined','local_interpolation','bracket_random'] else 1.2,
                    color=COLORS[m],label=NAMES[m],ls='--' if m in ['random','parameter_sobol'] else '-')
        ax.set(title=title,xlabel='Own paid-query budget',ylabel='Boundary discovery fraction (5 eligible sources)' if metric=='boundary_recall' else 'Mean nearest-descriptor distance over all candidates')
        ax.grid(alpha=.15);ax.axvline(64,color='#74838F',ls=':',lw=1)
    axes[0,0].legend(frameon=False,fontsize=8,ncol=2)
    offsets={'old_summary':(-36,-20),'stable':(-12,17),'quantitative':(-12,-18),'combined':(5,-13)}
    for m in active_methods:
        x=analysis['active']['64/distance'][m]['mean'];y=analysis['active']['64/boundary_recall'][m]['mean']
        axes[1,0].scatter(x,y,s=65,color=COLORS[m]);axes[1,0].annotate(NAMES[m],(x,y),xytext=offsets.get(m,(5,5)),
                        textcoords='offset points',fontsize=8,ha='right' if m in offsets and offsets[m][0]<0 else 'left',
                        arrowprops=dict(arrowstyle='-',lw=.5,color='#99A5AE') if m in offsets else None)
    axes[1,0].set(title='c  Two acquisition goals\nat the same budget (64)',xlabel='Descriptor distance (lower is better)',ylabel='Boundary discovery fraction',ylim=(.2,1.09));axes[1,0].grid(alpha=.15)
    groups=['bounds','reserve_effort','gap','followup','preference'];x=np.arange(len(groups))
    for i,m in enumerate(['stable','combined']):
        vals=[diagnostics['response_vs_reference'][g][m]['mean']/diagnostics['response_vs_reference'][g]['reference_zero']['mean'] for g in groups]
        axes[1,1].plot(x,np.maximum(vals,1e-8),'-o',color=COLORS[m],label=NAMES[m])
    axes[1,1].axhline(1,color='#74838F',ls='--',label='No-change reference')
    axes[1,1].set(xticks=x,xticklabels=['MPC bounds','Reserve effort','Recovery gap','Follow-up power','Preferences'],
                  ylabel='MAE / no-change MAE (lower is better; log scale)',title='d  Learned stationary outcomes\nrelative to a no-change prediction',yscale='log')
    axes[1,1].tick_params(axis='x',rotation=15);axes[1,1].legend(frameon=False);axes[1,1].grid(alpha=.15)
    fig.suptitle('Active acquisition: ranking quality and collection utility are separate tests',fontsize=17,fontweight='bold')
    for ext in ['png','svg']:fig.savefig(out/f'figure2_active_acquisition.{ext}',dpi=210,bbox_inches='tight')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);args=p.parse_args()
    root=Path(args.root).resolve();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    a=json.loads((root/'analysis.json').read_text('utf-8'));status=json.loads((root/'status.json').read_text('utf-8'))
    diagnostics=json.loads((root/'diagnostics.json').read_text('utf-8'))
    draw(a,out,diagnostics)
    ranking=a['ranking']['64/spearman'];capture=a['ranking']['64/top3_capture'];regret=a['ranking']['64/regret']
    rows=[[CN[m],f"{ranking[m]['mean']:.3f}",f"{capture[m]['mean']:.1%}",f"{regret[m]['mean']:.4f}",
           ' / '.join(f'{v:.3f}' for v in ranking[m]['seeds'].values())] for m in METHODS]
    effects=a['comparisons']['ranking/64/spearman']
    effectrows=[[CN[c['left']]+' − '+CN[c['right']],f"{c['difference']:+.3f}",f"[{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]",c['n_sources']] for c in effects]
    activerows=[[CN[m],f"{a['active']['64/boundary_recall'][m]['mean']:.1%}",f"{a['active']['64/distance'][m]['mean']:.6f}",
                 f"{a['active']['64/turnover_coverage'][m]['mean']:.1%}"] for m in METHODS+['bracket_random','random','parameter_sobol']]
    delta=next(c for c in effects if c['left']=='combined' and c['right']=='old_summary')
    diagnostic_rows=[]
    for m in ['old_summary_q','stable_q','stable_derived','combined_q','combined_derived','local_interpolation_q']:
        label={'stable_derived':'稳定后果预测→显式计算价值','combined_derived':'两者结合预测→显式计算价值'}.get(m,CN[m.replace('_q','')]+'Q评分' if m.endswith('_q') and m!='local_interpolation_q' else '局部物理插值')
        diagnostic_rows.append([label,f"{diagnostics['score_readouts']['spearman'][m]['mean']:.3f}",f"{diagnostics['score_readouts']['top3_capture'][m]['mean']:.1%}"])
    response_rows=[]
    group_cn={'bounds':'MPC功率上下界','reserve_effort':'参考储能恢复努力','gap':'恢复缺口','followup':'后续控制功率','preference':'后续偏好'}
    for g,values in diagnostics['response_vs_reference'].items():
        base=values['reference_zero']['mean'];model=values['stable']['mean']
        response_rows.append([group_cn[g],f'{model:.6f}',f'{base:.6f}',f'{model/base:.2f}'])
    direction='改善' if delta['difference']>0 else '下降'
    reliable='六个source的配对区间位于零以上，支持本缓存上的一致正向增益' if delta['ci95'][0]>0 else 'source配对区间仍包含零，不能将平均变化解释为稳定增益'
    text=f'''# 定量经验与稳定后果监督：零新仿真验证

本轮回答一个具体问题：采集器是否真正知道“下一次查询哪个动作更有信息价值”？复用162条已存在的分支、486组MPC测量；18条参考与端点分支为共同初始经验，144条中间动作构成付费查询池。新增物理仿真、LP求解、旧RL/预警器/专家训练全部为零。正式训练仅为24个新采集评分器，三种子42/43/44，CPU串行完成。

## 同历史排序：主要结果

四种模型读取相同Random历史，面对完全相同的未查候选。64次查询之后还有80条候选。排序真值、22维后果画像和信息奖励均沿用原定义。Spearman先在每个source内部计算，再平均三种子及source。top3捕获率衡量模型前三名实际获得的效用占真实最优前三名的比例；遗憾越小越好。

{table(['配置','Spearman','Top3真实效用捕获','Top3平均遗憾','三种子Spearman'],rows)}

两者结合相对旧摘要的排序{direction}{abs(delta['difference']):.3f}，95% source配对区间为[{delta['ci95'][0]:+.3f}, {delta['ci95'][1]:+.3f}]；{reliable}。局部插值以同一份已查询经验做对照，直接检验连续后果是否已经足以推算邻近动作的信息价值。

![同历史排序与归因](figure1_matched_ranking.png)

图1a–b比较同历史排序与前三项效用，淡色带为三个训练种子的范围；c展示每个source是否都改善，d给出source配对区间。128预算后每个source只剩2–3个候选，top3必然包括全部候选，因此b面板不展示这一失去区分度的点。

{table(['配对比较','Spearman变化','95% source CI','source数'],effectrows)}

## 主动采集：排序能否转化为覆盖

这一协议让各评分器根据自己的已查询历史采集128次，每次25%随机探索，其余按双critic最小评分选动作。各方法候选池、初始18条经验与source轮询保持相同。下表是新增64次查询时的结果；边界率只在五个确实存在边界的source上平均。画像距离的分母是完整144候选池，每项到最近已知画像的距离取平均，已查询候选的距离为0；它不是只在剩余80项上的平均。各source各含24项，先计算各自距离再平均。括区＋Random复用原冻结查询记录。

{table(['方法','相邻恢复边界定位率','固定候选池画像距离↓','转折形态覆盖'],activerows)}

![主动采集与固定后果学习](figure2_active_acquisition.png)

图2a–b分别考查定位边界和覆盖连续画像；两项任务不等价。c将64预算的两项结果放到一个坐标系，d以“不发生相对参考变化”的零预测为基准，检查稳定后果头的误差。纵坐标小于1表示优于零预测，恢复缺口则明显高于1，揭示稀少边界量尚未学好。组合模型的画像距离{a['active']['64/distance']['combined']['mean']:.6f}也没有优于旧摘要{a['active']['64/distance']['old_summary']['mean']:.6f}，不能把平均相关的小幅提高写成已实现的采集增益。

## 最重要的诊断：学后果与学一个Q值存在差别

在不增加任何训练的情况下，直接复用稳定后果头的已冻结预测，将预测的调度上下界与储能恢复努力变换成后果画像，再显式计算原信息奖励。恢复状态采用本缓存已核验吻合的1小时能量/功率必要边界，不采用该模型误差较大的直接缺口回归。排序相关从同一模型Q评分的0.295提高到0.514。配对提升0.219，95% source区间[0.022, 0.423]。与旧摘要相比提高0.265，区间[0.050, 0.529]。这一比较保持模型预测不变，改变物理投影与价值读出方式，支持：已经学到的一部分后果结构，没有被直接的价值评分充分利用。

{table(['冻结模型与读出方式','排序Spearman','Top3真实效用捕获'],diagnostic_rows)}

显式计算后，稳定模型前三项效用捕获也从57.3%升至71.3%，但这项提升的source区间仍跨零。0.514仍低于局部物理插值的0.736；该二级诊断尚未接入新的主动查询策略，不将它冒充主动采集胜出。

{table(['固定后果组','稳定模型MAE','零变化MAE','模型/零变化误差'],response_rows)}

上下界、恢复努力、后续控制和偏好的误差均低于零变化预测，说明稳定后果监督确实学习了行动对后续资源及调度的影响。恢复缺口恰好相反：大量零缺口与极少数小的正缺口使平均回归没有准确表达临界边界。三种子64预算的80条剩余候选中，存在正恢复缺口的分支数分别为{'、'.join(str(v) for v in diagnostics['remaining_gap_positive_by_seed'].values())}；端点是共同初始经验，剩余候选中的正例尤其稀少。这是下一步需要针对的学习对象，而非继续增加输入维数或网络宽度。

## 研究判断与下一步

第一，补齐输入带来部分source的排序改善，但既没有稳定的整体增益，也没有改善最优先查询的少数动作；空间和定量经验通道本身不足以解决问题。第二，稳定后果能被学习，其价值在“预测后果→依据当前已知经验计算信息价值”的分工中更清楚；把随经验库变化的价值直接交给一个Q标量，是当前最值得替换的环节。第三，局部插值利用相同经验即取得很强结果，说明当前两条剂量路径上应先保留可解释的物理计算，学习模块应争取补足插值不确定的转折与边界。

下一版可保持本轮稳定后果模型不变，在原缓存内先检验显式价值读出的主动采集，再把边界两侧的参考储能恢复余量作为成对判别任务。保留括区＋Random负责边界定位，局部插值负责平滑区段；仅在两者分歧或预测后果接近恢复边界时调用学习评分。该方向由本轮数据指向，尚未新增执行。

## 实现含义与可复核边界

空间输入补入各节点V/P/Q、逐站需求/期限/充放电能力、上次功率与可见到达预测；所有缩放只使用初始参考状态。定量经验保留同状态下已查询的相邻上下动作、最近动作的固定60维后果和22维画像，未查询的真实后果只能在选中并支付一次查询后进入训练。

稳定监督学习三个未来时刻（+1/+3/+5h）的MPC可行功率上下界、参考储能恢复所需努力、恢复缺口、后续控制请求和偏好变化。该60维量可重建全部162条分支的原画像，误差不超过1e-10。辅助头只读状态与动作，既不读取当前经验库，也不改变原奖励。四格使用相同双128层及相同参数总数；旧配置把新增输入通道置零。

局部插值只使用同状态同动作族已查询的相邻剂量，并依据1小时、1MW充电率和0.95效率的资源边界重建恢复状态。本缓存中这一必要能量下界与已有LP缺口恰好一致，因此它是强物理对照；它并不替代一般网络约束下的联合可恢复性检验。本轮是相同六个历史source上的开发复用验证，三种子不是十八个独立物理source。

## 执行与数据

运行目录：`{root}`。全部候选评分、真值、固定后果预测、各source统计和checkpoint已保存；`source_records.json`和`analysis.json`可直接复算。原始输入与旧代码/旧checkpoint共{status['frozen_files']}项文件哈希前后核对一致。

```powershell
Set-Location -LiteralPath 'D:\\博士资料包\\科研\\科研\\AAAI_RL\\tarp_37_information_acquisition'
powershell -ExecutionPolicy Bypass -File .\\scripts\\run_acquisition_evidence_v3.ps1 -Mode formal -RunId evidence_formal_s42_v1 -Resume
```

相同RunId加`-Resume`用于核验和跳过已完成任务，不能用同一RunId更改配置。新完整重复实验请另设RunId。
'''
    (out/'report.md').write_text(text,encoding='utf-8')
    html=markdown.markdown(text,extensions=['tables','fenced_code'])
    (out/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>定量经验与稳定后果监督</title><style>body{max-width:1220px;margin:36px auto;padding:0 24px;font:17px/1.85 "Microsoft YaHei",sans-serif;color:#263747}h1{font-size:29px}h2{font-size:23px;margin-top:36px}img{width:100%}table{width:100%;border-collapse:collapse;font-size:15px}th,td{padding:9px;border-bottom:1px solid #dce5e9;text-align:left}th{background:#edf4f6}pre{background:#f3f6f7;padding:18px;overflow:auto}code{font-size:13px}</style><body>'+html+'</body></html>',encoding='utf-8')
    print(str(out/'index.html'))


if __name__=='__main__':main()
