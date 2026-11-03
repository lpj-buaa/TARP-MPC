from pathlib import Path
import argparse,html,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tarp_boundary.io import read,write

LABELS={'ppo':'Original PPO','rank':'Ranking (R)','correct':'Conditional correction (C)',
    'rank_correct':'R + C','correct_fixed':'Fixed correction','rank_correct_fixed':'R + fixed correction',
    'local':'Local interpolation','bracket_random':'Bracket + Random','random':'Random','sobol':'Sobol'}
COLORS={'ppo':'#64748b','rank':'#c46d39','correct':'#12867c','rank_correct':'#215d9b',
    'correct_fixed':'#90b9af','rank_correct_fixed':'#9c89b8','local':'#aa4964','bracket_random':'#b89a3f'}


def report(root):
    root=Path(root);out=root/'report';out.mkdir(parents=True,exist_ok=True)
    a=read(root/'analysis.json');p=read(root/'run_protocol.json');dev=a['summary']['development_test']
    diag=a['diagnostic_regression']['development_test'];rank=a['diagnostic_ranking']['development_test']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,
        'axes.labelcolor':'#243746','text.color':'#243746','axes.titleweight':'bold','svg.fonttype':'none'})
    fig,axes=plt.subplots(3,2,figsize=(13,13),layout='constrained')
    for m in ['ppo','rank','correct','rank_correct','local','bracket_random']:
        axes[0,0].plot([8,16,24],[dev[m][str(b)]['utility']['mean'] for b in [8,16,24]],'o-',label=LABELS[m],color=COLORS[m],lw=1.5)
    axes[0,0].set(title='a  Equal-query-budget acquisition',xlabel='Additional six-hour branch queries',ylabel='Terminal information utility U',xticks=[8,16,24])
    axes[0,0].legend(fontsize=7,ncol=2)
    arms=['ppo','rank','correct_fixed','correct','rank_correct_fixed','rank_correct']
    axes[0,1].barh(np.arange(6),[dev[m]['8']['utility']['mean'] for m in arms],color=[COLORS[m] for m in arms])
    axes[0,1].set(yticks=np.arange(6),yticklabels=[LABELS[m] for m in arms],xlabel='U after eight queries',title='b  What changes without adding parameters?')
    for i,m in enumerate(arms):axes[0,1].text(dev[m]['8']['utility']['mean']+.003,i,f"{dev[m]['8']['utility']['mean']:.3f}",va='center',fontsize=8)
    axes[0,1].set_xlim(0,min(1,max(dev[m]['8']['utility']['mean'] for m in arms)+.09))
    for m,lab in [('base','Original prior'),('correct_fixed','Fixed correction'),('correct','Learned correction'),('local','Local interpolation')]:
        axes[1,0].plot([0,4,8],[diag[m][str(h)]['mae']['mean'] for h in [0,4,8]],'o-',label=lab,color=COLORS.get(m,'#64748b'))
    axes[1,0].set(title='c  Predicting only unqueried consequences',xlabel='Observed additional branches (shared histories)',ylabel='Normalized continuous MAE',xticks=[0,4,8])
    axes[1,0].legend(fontsize=8)
    methods=['ppo','rank','correct','rank_correct']
    for i,m in enumerate(methods):
        value=rank['policy'][m]['mean'];axes[1,1].barh(i,value,color=COLORS[m])
        axes[1,1].text(value+(.01 if value>=0 else -.01),i,f'{value:.3f}',ha='left' if value>=0 else 'right',va='center')
    local=rank['response_readout']['local']['mean']
    axes[1,1].axvline(local,color=COLORS['local'],ls='--',label=f'Local readout: {local:.3f}')
    axes[1,1].set(yticks=np.arange(4),yticklabels=[LABELS[m] for m in methods],xlabel='Spearman rho: score vs true immediate gain',title='d  Does the actor learn information ranking?')
    axes[1,1].legend(fontsize=8)
    sources=list(dev['ppo']['8']['utility']['sources'])
    for j,m in enumerate(['rank','correct','rank_correct']):
        gain=[100*(dev[m]['8']['utility']['sources'][s]-dev['ppo']['8']['utility']['sources'][s]) for s in sources]
        axes[2,0].bar(np.arange(len(sources))+(j-1)*.22,gain,width=.2,label=LABELS[m],color=COLORS[m])
    axes[2,0].set(xticks=np.arange(len(sources)),xticklabels=[s.replace('_','\n') for s in sources],ylabel='U change vs PPO (percentage points)',title='e  Paired gains in each development source')
    axes[2,0].axhline(0,color='#64748b',lw=.8);axes[2,0].legend(fontsize=8)
    source=Path(p['source'])
    for parent,lab,col in [(source,'Original','#64748b'),(root,'Conditional','#12867c')]:
        for split,style in [('train','-'),('validation','--')]:
            histories=[read(parent/f'response_s{s}/fit.json')['history'] for s in p['seeds']]
            length=max(len(h) for h in histories)
            means=[np.mean([h[e][split]['mae'] for h in histories if e<len(h)]) for e in range(length)]
            axes[2,1].plot(np.arange(1,length+1),means,style,color=col,label=f'{lab}: {split}')
    axes[2,1].set(title='f  Training and tuning errors (three seeds)',xlabel='Response-model epoch',ylabel='Unconditioned MAE')
    axes[2,1].legend(fontsize=8)
    for ax in axes.flat:ax.grid(axis='y',alpha=.14);ax.set_axisbelow(True)
    fig.suptitle('Aligning acquisition with information value, at fixed network capacity',fontsize=16,fontweight='bold')
    fig.savefig(out/'alignment_ablation.png',dpi=200);fig.savefig(out/'alignment_ablation.svg');plt.close(fig)
    svg=out/'alignment_ablation.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    table=[]
    for m in ['ppo','rank','correct_fixed','correct','rank_correct_fixed','rank_correct','local','bracket_random','random','sobol']:
        table.append('<tr><td>'+LABELS[m]+'</td>'+''.join(f'<td>{dev[m][str(b)]["utility"]["mean"]:.4f}</td>' for b in [8,16,24])+'</tr>')
    effects=''.join(f'<tr><td>{name}</td>'+''.join(f'<td>{values[str(b)]["mean"]*100:+.2f}</td>' for b in [8,16,24])+'</tr>' for name,values in a['contrasts'].items())
    md=['# 固定容量信息对齐复用验证','',
        '本轮以相同的完整情境、归一化、角色和种子，区分信息价值排序监督R、观测条件化后果学习C，以及无新增训练目标的固定残差修正。',
        f"原后果模型{p['response_parameters']}参数，原采集策略{p['policy_parameters']}参数，均未扩容。训练10个source、调参4个source，开发留出2个source；原正式confirm未使用。",'',
        '一次预算查询是同一初始情境下一个新候选动作的完整六小时MPC接续，返回三个后续时点的资源/恢复测量。每方法预算8意味着在共同3条初始分支以外再查8条，绝不是八次与一次相比。银行完整池有32条动作，本轮回放已封存答案，没有新物理仿真。','',
        '## 同预算结果','', '|方法|8次|16次|24次|','|---|---:|---:|---:|']
    for m in dev:md.append('|'+LABELS[m]+'|'+'|'.join(f"{dev[m][str(b)]['utility']['mean']:.4f}" for b in [8,16,24])+'|')
    md+=['','## 机制和监督','',
        'R直接监督策略对边际信息价值的排序，标签仅由训练银行产生；C用已经观测的真实后果残差修正未查询候选，并在训练阶段模拟这种条件化任务。C-fixed只应用同一规则，帮助识别条件化学习的额外作用。排序rho、后果MAE和最终U分开报告；MAE只统计未查询动作，避免已知值回填造成虚假收益。','',
        '## 计算口径','',str(a['training']),'',
        'U沿用后果画像覆盖与恢复边界双端检出的效用，它不是复合风险召回率，也不是连续动作空间全覆盖证明。source统计先在同source情境与训练种子内平均，再对source等权；这轮两条开发留出source用于寻找机制，未改训练配方。','',
        '## 相关研究与本文定位','',
        '[Deep Adaptive Design (ICML 2021)](https://proceedings.mlr.press/v139/foster21a.html)学习基于历史实验结果的自适应设计策略；[Attentive Neural Processes (ICLR 2019)](https://arxiv.org/abs/1901.05761)学习基于观测上下文的条件预测。我们的方向是将持续MPC调度中后续资源范围、转折和恢复边界直接定义为采集目标，在固定容量下检验任务对齐与观测修正的作用。一般的排序学习与观测条件化不作首次提出的主张。']
    (out/'findings.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>信息对齐 V5</title><style>
    body{font:17px/1.8 'Microsoft YaHei',sans-serif;color:#253746;background:#f3f6f8;margin:0}main{max-width:1280px;margin:32px auto;padding:40px;background:white}h1,h2{color:#183d56}img{width:100%}table{border-collapse:collapse;width:100%;margin:24px 0}td,th{padding:10px;border-bottom:1px solid #ccd6df;text-align:right}td:first-child,th:first-child{text-align:left}pre{white-space:pre-wrap;background:#f5f8fa;padding:16px}a{color:#215d9b}</style><main>
    <h1>信息价值排序与观测后修正：固定容量消融</h1>
    <p>所有方法使用同一数据银行、同一源划分与同一查询预算。原采集保持暂停，旧RL和预警器未训练。</p>
    <img src="alignment_ablation.png" alt="Fixed capacity six-panel ablation">
    <h2>同预算信息采集</h2><p>U为后果覆盖与恢复边界检出的效用，按source宏平均。</p>
    <table><tr><th>方法</th><th>8次查询</th><th>16次查询</th><th>24次查询</th></tr>'''+''.join(table)+'''</table>
    <h2>收益来自哪里</h2><p>表内为U变化的百分点；固定规则对照将非参数修正和条件化学习拆开。</p>
    <table><tr><th>配对比较</th><th>8次</th><th>16次</th><th>24次</th></tr>'''+effects+'''</table>
    <h2>完整方法与数据口径</h2><pre>'''+html.escape('\n'.join(md))+'''</pre></main></html>'''
    if (root/'readout_probe.json').exists():
        probe=read(root/'readout_probe.json')
        labels={'static':'原后果模型＋同目标贪心','fixed':'固定残差修正＋同目标贪心','learned':'条件化修正＋同目标贪心'}
        probe_rows=''.join('<tr><td>'+labels[m]+'</td>'+''.join(f'<td>{probe["summary"][m][str(b)]["utility"]["mean"]:.4f}</td>' for b in [8,16,24])+'</tr>' for m in labels)
        page=page.replace('<h2>完整方法与数据口径</h2>',
            '<h2>追加机制诊断：后果信息能否被利用</h2><p>以下为冻结模型的事后读出检查，所有可运行方法依然只看已查询答案，无新增训练或仿真。它不用于选择本轮训练配方。</p>'+
            '<table><tr><th>读出方式</th><th>8次</th><th>16次</th><th>24次</th></tr>'+probe_rows+'</table>'+
            '<p>另外，在每个策略的72个共同历史决策点，将候选后果全部清零或替换为完整真值，原PPO、R与R+C的首选动作均未改变。这里真值输入仅用于诊断，不是可部署对照。这表明当前策略对候选后果缺乏有效利用，预测变准尚未充分转化成查询选择。</p>'+
            '<h2>完整方法与数据口径</h2>')
    (out/'index.html').write_text(page,encoding='utf-8')
    with (out/'evaluation_rows.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['method','seed','source_id','key','role','budget','utility','distance','boundary_recall','unknown_queries']
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(read(root/'evaluation.json')['rows'])
    write(out/'report_manifest.json',dict(development_sources=2,original_confirm_used=0,figure='alignment_ablation.svg',training=a['training']))
    print(str(out/'index.html'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);report(p.parse_args().root)
