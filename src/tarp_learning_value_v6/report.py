from pathlib import Path
import argparse,html,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tarp_boundary.io import read

LABELS={'value_greedy':'Learning-value greedy','value_ppo':'Learning-value + PPO','local':'Local interpolation',
    'bracket_random':'Bracket + Random','random':'Random','hybrid':'PPO + bracket rule','greedy_bracket':'Greedy + bracket rule','initial_only':'Initial branches only'}
COLORS={'value_greedy':'#177f79','value_ppo':'#235f9e','local':'#a44e6b','bracket_random':'#b18c31',
    'random':'#84929e','hybrid':'#8663ac','greedy_bracket':'#bd7853','initial_only':'#555555'}


def make_report(root):
    root=Path(root);out=root/'report';out.mkdir(parents=True,exist_ok=True)
    e=read(root/'evaluation.json');test=e['summary']['test'];protocol=read(root/'run_protocol.json')
    exposure=read(root/'exposure_protocol.json');methods=[m for m in ['value_greedy','value_ppo','local','bracket_random','random','hybrid','greedy_bracket'] if m in test]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(13,10),layout='constrained')
    initial=test['initial_only']['0']['task_error']['mean']
    for m in methods:
        axes[0,0].plot([0,8,16,24],[initial]+[test[m][str(b)]['task_error']['mean'] for b in [8,16,24]],'o-',color=COLORS[m],label=LABELS[m])
    axes[0,0].set(title='a  Do selected actions improve the learned predictor?',xlabel='Additional queries per training context',ylabel='Held-out worst-component prediction error',xticks=[0,8,16,24])
    axes[0,0].legend(fontsize=7)
    coverage=np.array([[test[m]['8'][f'coverage_{eps}']['mean'] for eps in [.1,.25,.5,1.]] for m in methods])
    image=axes[0,1].imshow(coverage,cmap='YlGnBu',vmin=0,vmax=1,aspect='auto')
    for j in range(coverage.shape[0]):
        for k in range(4):axes[0,1].text(k,j,f'{coverage[j,k]:.0%}',ha='center',va='center',color='white' if coverage[j,k]>.55 else '#263744')
    axes[0,1].set(xticks=range(4),xticklabels=['0.10','0.25','0.50','1.00'],yticks=range(len(methods)),yticklabels=[LABELS[m] for m in methods],
        xlabel='Allowed joint prediction error',title='b  Learned consequence-domain coverage (budget 8)')
    fig.colorbar(image,ax=axes[0,1],label='Fraction with error within tolerance',shrink=.7)
    sources=e['test_sources']
    for j,b in enumerate(['8','16']):
        g=test['value_greedy'][b]['task_error']['sources'];p=test['value_ppo'][b]['task_error']['sources']
        axes[1,0].bar(np.arange(len(sources))+(j-.5)*.3,[p[s]-g[s] for s in sources],width=.28,label=f'Budget {b}',color=['#235f9e','#74b5be'][j])
    axes[1,0].axhline(0,color='#666',lw=.8);axes[1,0].legend(fontsize=8)
    axes[1,0].set(xticks=range(len(sources)),xticklabels=[s.replace('_','\n') for s in sources],ylabel='PPO error minus greedy error (lower is better)',title='c  Does RL add value in each independent source?')
    groups=['bounds','reserve_effort','gap','followup','preference']
    def group_mean(method,group):
        rows=[r for r in e['rows'] if r['role']=='test' and r['method']==method and r['budget']==16]
        return np.mean([np.mean([r['groups'][group] for r in rows if r['source_id']==s]) for s in sources])
    reference=np.array([max(group_mean('random',g),1e-9) for g in groups])
    ratios=np.array([(np.array([group_mean(m,g) for g in groups])/reference-1)*100 for m in methods])
    im=axes[1,1].imshow(ratios,cmap='BrBG_r',vmin=-80,vmax=80,aspect='auto')
    for j in range(len(methods)):
        for k in range(5):axes[1,1].text(k,j,f'{ratios[j,k]:+.0f}%',ha='center',va='center',fontsize=8,color='white' if abs(ratios[j,k])>55 else '#263744')
    axes[1,1].set(xticks=range(5),xticklabels=['Bounds','Reserve','Gap','Dispatch','Preference'],yticks=range(len(methods)),
        yticklabels=[LABELS[m] for m in methods],title='d  Which physical outcomes improve? (budget 16)')
    fig.colorbar(im,ax=axes[1,1],label='MAE change vs Random (%)',shrink=.7)
    for ax in [axes[0,0],axes[1,0]]:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('Active learning of action consequences: fixed data and network capacity',fontsize=15,fontweight='bold')
    fig.savefig(out/'learning_value.png',dpi=200);fig.savefig(out/'learning_value.svg');plt.close(fig)
    svg=out/'learning_value.svg';svg.write_text('\n'.join(x.rstrip() for x in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    lines=['# 行动后果主动学习复用验证','',
        f"本轮复用现有封存数据，神经网络容量固定。训练/调参/独立测试/开发source数分别为{len(exposure['roles']['train'])}/{len(exposure['roles']['tune'])}/{len(exposure['roles']['test'])}/{len(exposure['roles']['development'])}。初始模型只见每训练情境的3条共同分支；每情境另有24条可查询、5条固定奖励/审计分支。",'',
        '独立测试为历史cal source，本轮未用于训练、奖励或模型选择。测试source上的所有32条动作均只作最终评价，测试时后果预测器不适应、不训练。该结果不冒充新仿真或全新confirm。','',
        '## 合并选样训练后的独立测试','',
        '|方法|8次任务误差|16次任务误差|24次任务误差|','|---|---:|---:|---:|']
    for m in methods:lines.append('|'+LABELS[m]+'|'+'|'.join(f"{test[m][str(b)]['task_error']['mean']:.5f}" for b in [8,16,24])+'|')
    lines+=['',f'仅初始分支的任务误差：{initial:.5f}。','',
        '任务误差为60维连续后果按训练初始尺度归一化的最大绝对误差，以及3个恢复失败分数误差的最大值；失败分数不宣称已校准为真实概率。覆盖定义为上述联合误差不超过指定epsilon的动作比例。','',
        '所有方法选出的样本交给同一容量、同一训练规程的预测器；主要比较预算8/16。预算24时各方法样本集合完全一致，指纹核验并复用相同训练结果。','',
        '## RL与规则混合','',str(e['hybrid']),'',
        '混合由调参结果触发，比例与规则提前固定；独立测试不决定是否换比例或再训练。','',
        '## 能力与计算边界','',e['target'],'',str(e['training']),'',
        '原物理环境、容量、阈值、旧RL与专家冻结；本轮物理仿真新增0。episode中通过现有输出层快速学习产生实际预测误差奖励；表中主结果则来自合并选样后全参数训练的模型，二者没有混为同一个指标。','',
        '## 执行命令','',
        "```powershell\nSet-Location -LiteralPath 'D:\\博士资料包\\科研\\科研\\AAAI_RL\\tarp_37_information_acquisition'\npowershell -ExecutionPolicy Bypass -File .\\scripts\\run_learning_value_v6.ps1 `\n  -RunId "+root.name+' -Mode '+protocol['mode']+' -Device '+protocol['device']+' -Resume\n```']
    diagnostic=read(root/'diagnostics.json')
    lines+=['','## 信息价值与输入使用核验','',str(diagnostic['value_summary']),'',
        '信息评分、候选后果与已查后果通道分别扰动，避免把强制使用信息分数等同于原始后果通道敏感。oracle信息仅在冻结后作诊断。','',
        str(diagnostic['sensitivity_summary']),'',
        'PPO最佳批次（0为未训练的价值贪心起点）：'+str(diagnostic['ppo_best_batches']),'',
        '## 奖励面快速学习收益（训练面，区别于独立测试）','',str(e['fast_learning_summary'])]
    text='\n'.join(lines)+'\n';(out/'findings.md').write_text(text,encoding='utf-8')
    import markdown
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>行动后果主动学习</title><style>body{font:17px/1.85 'Microsoft YaHei',sans-serif;background:#f0f4f7;color:#243746;margin:0}main{max-width:1300px;margin:30px auto;background:white;padding:40px}img{width:100%}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ccd6df;text-align:right}td:first-child,th:first-child{text-align:left}pre{white-space:pre-wrap;background:#f5f7f9;padding:16px}h1,h2{color:#164d66}</style><main>'''+markdown.markdown(text,extensions=['tables','fenced_code'])+'<h2>学习收益、覆盖与物理后果组合图</h2><img src="learning_value.png" alt="Learning value validation"></main></html>'
    (out/'index.html').write_text(page,encoding='utf-8')
    fields=['method','budget','seed','key','source_id','role','task_error','mae','delta_mae','direction_accuracy','recovery_failure_recall','recovery_failure_fpr']
    with (out/'rows.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(e['rows'])
    print(str(out/'index.html'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);make_report(p.parse_args().root)
