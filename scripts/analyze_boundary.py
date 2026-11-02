"""Source-level acquisition comparisons and native mechanism atlas."""
import argparse,json
from pathlib import Path
import numpy as np


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',default='boundary_formal_s42_v2');args=parser.parse_args()
    project=Path(__file__).resolve().parents[1];root=project/'artifacts/boundary_v2'/args.run_id
    bundle=read(root/'bundle.json');inventory=read(root/'inventory.json');complete=read(root/'acquisition_complete.json');assert complete['complete']
    methods=['random','sobol','error_td3','boundary_reward_only','bracket_random','boundary_td3']
    names={'random':'Random','sobol':'Sobol','error_td3':'Error critics','boundary_reward_only':'Boundary reward only','bracket_random':'Bracket + random','boundary_td3':'Boundary critics + pairing'}
    colors=dict(zip(methods,['#8C99A3','#39867E','#4B77A4','#927BA4','#CB9A48','#B55853']))
    runs=[read(root/'acquisition'/f'{m}_s{s}.json') for m in methods for s in [42,43,44]]
    table=[]
    for budget in [64,128]:
        for method in methods:
            records=[next(x for x in r['evaluation'] if x['budget']==budget) for r in runs if r['method']==method]
            mean=lambda k:float(np.mean([x[k] for x in records if x[k] is not None])) if any(x[k] is not None for x in records) else None
            table.append(dict(budget=budget,method=method,**{k:mean(k) for k in ['boundary_recall','boundary_pairs_found','boundary_pairs_total','boundary_contexts_found','boundary_contexts_total','boundary_bracket_width','distance','turnover_coverage','joint','unknown']}))
    ci={}
    for base in [m for m in methods if m!='boundary_td3']:
        samples={}
        for method in [base,'boundary_td3']:
            by={}
            for r in runs:
                if r['method']!=method:continue
                for x in next(e for e in r['evaluation'] if e['budget']==64)['sources']:
                    if x['boundary_recall'] is not None:by.setdefault(x['source_id'],[]).append(x['boundary_recall'])
            samples[method]={k:float(np.mean(v)) for k,v in by.items()}
        keys=sorted(set(samples[base])&set(samples['boundary_td3']));d=np.array([samples['boundary_td3'][k]-samples[base][k] for k in keys])
        boot=np.random.default_rng(42).choice(d,(2000,len(d)),replace=True).mean(1)
        ci[base]=dict(gain=float(d.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),independent_sources=len(keys))
    raw={x['id']:read(x['path']) for x in inventory};physics=[];known_hours=0;unverified=0;extra=0;shed=0.;clipped=0;guards=0
    for ident,r in raw.items():
        for h,row in enumerate(r['rows']):
            if row.get('physical_valid'):
                assert row['max_bus_residual_mva']<1e-5 and row['generator_bound_excess']<1e-6
                assert min(row['voltage'])>=.93-1e-6 and max(row['voltage'])<=1.07+1e-6
                assert row['max_line_loading_ratio']<=1.+1e-6
                assert min(row['ess_energy_mwh'])>=.2-1e-7 and max(row['ess_energy_mwh'])<=1.8+1e-7
                assert row['battery_balance_error_kwh']<1e-5 and row['ess_balance_error_mwh']<1e-7
                known_hours+=1
            else:unverified+=1
            if row.get('normal_regulation',{}).get('status')=='voltage_ok_other_boundary':extra+=1
            shed+=row.get('non_ev_shed_mwh',0.)
            if h==0:
                clipped+=bool(np.max(np.abs(np.array(row['candidate_request_mw'])-row['bounded_request_mw']))>1e-7)
                guards+=bool(row.get('forecast_allocation_guard',{}).get('changed'))
    allrows=bundle['initial']+[dict(p,**bundle['answers'][p['id']]) for p in bundle['public']]
    support=dict(total_branches=len(inventory),initial=len(bundle['initial']),candidates=len(bundle['public']),
        known_physical_hours=known_hours,unverified_hours=unverified,unknown_descriptors=sum(r['descriptor'] is None for r in allrows),
        additional_active_recourse_hours=extra,non_ev_shed_mwh=shed,first_action_bounded=clipped,forecast_guard_changed=guards,
        recovery_support=[dict(hour=h+1,recoverable=sum(r['recoverable'] is not None and r['recoverable'][j] for r in allrows),
            nonrecoverable=sum(r['recoverable'] is not None and not r['recoverable'][j] for r in allrows)) for j,h in enumerate([0,2,4])])
    result=dict(complete=True,run_id=args.run_id,primary_budget=64,table=table,source_ci=ci,support=support,
        scope=bundle['scope'],forecaster_training_calls=0,old_models_training_calls=0)
    (root/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    out=project/'docs/boundary_acquisition_2026-11-02';out.mkdir(parents=True,exist_ok=True)
    fig,axs=plt.subplots(2,2,figsize=(13,9.3));fig.subplots_adjust(left=.08,right=.97,top=.87,bottom=.12,wspace=.28,hspace=.38)
    fig.suptitle('Acquiring both sides of downstream recovery boundaries',x=.08,ha='left',fontsize=18,weight='bold')
    fig.text(.08,.92,'6 fixed initial states | 18 initial + 144 shared candidate branches | 3 seeds | no warning-model training',color='#6c7b86')
    for method in methods:
        subset=[r for r in runs if r['method']==method];budgets=[e['budget'] for e in subset[0]['evaluation']]
        for ax,field in [(axs[0,0],'boundary_recall'),(axs[0,1],'boundary_bracket_width'),(axs[1,0],'distance')]:
            values=np.array([[e[field] for e in r['evaluation']] for r in subset],float)
            ax.plot(budgets,values.mean(0),color=colors[method],label=names[method],marker='o',ms=3)
            ax.fill_between(budgets,np.nanmin(values,axis=0),np.nanmax(values,axis=0),color=colors[method],alpha=.08)
    axs[0,0].set(title='a  Exact adjacent-dose boundaries recovered',ylabel='Mean within-source recovery fraction',xlabel='Added answers');axs[0,0].legend(frameon=False,fontsize=7)
    axs[0,1].set(title='b  Remaining boundary uncertainty',ylabel='Mean dose bracket width (lower is better)',xlabel='Added answers')
    axs[1,0].set(title='c  Coverage of distinct downstream profiles',ylabel='Mean remaining profile distance',xlabel='Added answers')
    points=[r for r in table if r['budget']==64]
    for i,row in enumerate(points):
        vals=[next(e for e in r['evaluation'] if e['budget']==64)['boundary_pairs_found'] for r in runs if r['method']==row['method']]
        axs[1,1].scatter([i-.07,i,i+.07],vals,color=colors[row['method']],s=30)
        axs[1,1].plot([i-.2,i+.2],[np.mean(vals)]*2,color=colors[row['method']],lw=2)
    axs[1,1].set(title='d  Matched-budget mechanism comparison',ylabel='Distinct adjacent boundary pairs found at 64',xticks=range(6),xticklabels=['Random','Sobol','Error','Reward\nonly','Bracket\nrandom','Hybrid'])
    for ax in axs.ravel():ax.grid(axis='y',color='#e8edf0');ax.set_axisbelow(True)
    fig.text(.08,.04,'Bands/dots: all three seeds. Boundaries concern one-hour reserve recovery within each state’s own +5% business budget; not new safety limits.',fontsize=8,color='#6c7b86')
    save(fig,out,'comparison',plt)
    # Full dose atlas: no handpicked actions or states.
    keys=sorted({r['key'] for r in allrows});gmax=max((max(r['gap_mwh']) for r in allrows if r['descriptor'] is not None),default=1.)
    fig,axs=plt.subplots(3,2,figsize=(13,10),sharex=True,sharey=True);fig.subplots_adjust(left=.12,right=.88,top=.87,bottom=.13,hspace=.2,wspace=.12)
    fig.suptitle('A native-simulation atlas of reserve recovery',x=.12,ha='left',fontsize=18,weight='bold')
    fig.text(.12,.92,'Every dose, every source | color: minimum reserve gap | outlines: adjacent-dose recovery transition',color='#6c7b86')
    for j,h in enumerate([1,3,5]):
        for col,family in enumerate(['throttle','release']):
            ax=axs[j,col];grid=np.full((len(keys),14),np.nan)
            for row,key in enumerate(keys):
                group=[r for r in allrows if r['key']==key and r['family'] in [family,'reference']]
                for r in group:
                    if r['descriptor'] is not None:grid[row,round(r['dose']*13)]=r['gap_mwh'][j]
                for k in range(13):
                    if np.isfinite(grid[row,k:k+2]).all() and (grid[row,k]<=1e-6)!=(grid[row,k+1]<=1e-6):
                        ax.plot([k+.5,k+.5],[row-.45,row+.45],color='#1e3548',lw=2)
            im=ax.imshow(np.ma.masked_invalid(grid),vmin=0,vmax=max(gmax,.01),cmap='YlOrRd',aspect='auto')
            ax.set(yticks=range(len(keys)),yticklabels=[x[:10] for x in keys],xticks=[0,3,6,9,13],xticklabels=['0','3/13','6/13','9/13','1'])
            ax.set_title(f'{family.capitalize()} · decision at +{h} h',fontsize=11)
            if j==2:ax.set_xlabel('Intervention dose between reference and endpoint')
    cax=fig.add_axes([.91,.24,.015,.5]);fig.colorbar(im,cax=cax,label='Minimum reserve recovery gap (MWh)')
    fig.text(.12,.055,'Zero: matched reference reserves can be recovered in the next hour. Missing/unknown is masked. Native capacities and risk thresholds unchanged.',fontsize=8,color='#6c7b86')
    save(fig,out,'recovery_atlas',plt)
    fig,axs=plt.subplots(2,2,figsize=(13,9.6));fig.subplots_adjust(left=.12,right=.89,top=.88,bottom=.12,wspace=.16,hspace=.42)
    fig.suptitle('How intervention classes reshape later dispatch options',x=.12,ha='left',fontsize=17,weight='bold')
    fig.text(.12,.925,'All six sources and all 12 interior doses per family | fixed nominal power normalization',color='#6c7b86')
    tiles=[]
    for family in ['throttle','release']:
        tile=np.array([np.median([np.array(r['width_delta']).ravel()*100 for r in allrows if r['key']==key and r['family']==family and 0<r['dose']<1 and r['descriptor'] is not None],axis=0) for key in keys])
        tiles.append(tile)
    vmax=max(1.,max(float(np.nanmax(np.abs(t))) for t in tiles))
    for col,family in enumerate(['throttle','release']):
        ax=axs[0,col];im=ax.imshow(tiles[col],cmap='RdBu_r',vmin=-vmax,vmax=vmax,aspect='auto')
        for i in range(len(keys)):
            for j in range(12):ax.text(j,i,f'{tiles[col][i,j]:+.1f}',ha='center',va='center',fontsize=7,color='white' if abs(tiles[col][i,j])>.6*vmax else '#263a49')
        for k in [3.5,7.5]:ax.axvline(k,color='white',lw=2)
        ax.set(yticks=range(len(keys)),yticklabels=[x[:10] for x in keys],xticks=range(12),
            xticklabels=[f'{d}\n+{h}h' for h in [1,3,5] for d in ['S','F','E','P']],title=f'{family.capitalize()}: median change across doses')
        ax=axs[1,col]
        for h,c in enumerate(['#4B77A4','#CB9A48','#39867E']):
            trajectories=[]
            for key in keys:
                rr=[r for r in allrows if r['key']==key and r['family'] in [family,'reference']]
                trajectories.append([next((r['width_delta'][h][2]*100 for r in rr if abs(r['dose']-j/13)<1e-8 and r['descriptor'] is not None),np.nan) for j in range(14)])
            trajectories=np.array(trajectories);dose=np.arange(14)/13
            ax.plot(dose,np.nanmedian(trajectories,axis=0),color=c,marker='o',ms=3,label=f'Decision +{[1,3,5][h]} h')
            ax.fill_between(dose,np.nanquantile(trajectories,.25,axis=0),np.nanquantile(trajectories,.75,axis=0),color=c,alpha=.12)
        ax.axhline(0,color='#8c99a3',lw=1);ax.axhline(1,color='#d9e3e8',ls='--',lw=.8);ax.axhline(-1,color='#d9e3e8',ls='--',lw=.8)
        ax.set(title=f'{family.capitalize()}: ESS range response',xlabel='Intervention dose',ylabel='Range change / nominal scale (%)')
        ax.legend(frameon=False,fontsize=8);ax.grid(axis='y',color='#edf2f4')
    cax=fig.add_axes([.92,.56,.015,.3]);fig.colorbar(im,cax=cax,label='Range change / nominal scale (%)')
    fig.text(.12,.035,'S/F/E/P = SCS/FCS/ESS/PCC. Positive: expansion; negative: contraction. Curves: source median and IQR. Directional supports, not feasible volumes.',fontsize=8,color='#6c7b86')
    save(fig,out,'dispatch_shapes',plt)
    # Preserve every actual boundary pair as reproducible native/measurement IDs.
    import sys
    sys.path.insert(0,str(project/'src'))
    from tarp_boundary.metrics import adjacent_boundaries,expanded
    pairs=sorted({(a,b) for a,b,h in adjacent_boundaries(expanded(allrows))})
    (root/'boundary_pairs.json').write_text(json.dumps([dict(left=a,right=b) for a,b in pairs],indent=2),encoding='utf-8')
    header='| 方法 | 相邻边界定位率 | 找到的边界对 | 剩余括区宽度↓ | 画像覆盖距离↓ |\n|---|---:|---:|---:|---:|\n'
    tablemd=''.join(f"| {names[r['method']]} | {r['boundary_recall']:.1%} | {r['boundary_pairs_found']:.1f}/{r['boundary_pairs_total']:.0f} | {r['boundary_bracket_width']:.3f} | {r['distance']:.5f} |\n" for r in table if r['budget']==64)
    text=f'''# 后续调度范围与恢复边界：信息采集结果

本轮冻结容量、风险阈值、原物理代码、旧RL和全部预警器。新增学习仅发生在信息采集器，接续仍为动态偏好有限信息MPC。目标是使采集经验覆盖不同后续调度后果，并把同状态下恢复边界定位到相邻剂量格。

## 主要结果：每臂64次新增答案

{header}{tablemd}

![采集对照](comparison.png)

表为三seed均值；共同初始18条，候选144条。边界对是同source、同动作族、相邻lambda格的两条真实分支，至少一个未来时点的可恢复状态不同。两端各计一次查询；初始0和1只粗括边界，不算已经定位相邻边界。128查询的饱和结果、每个source和所有seed保留于analysis.json及acquisition目录。

## 恢复边界的物理与业务含义

每个候选实际推演六小时，保留+1、+3、+5小时的真实决策视图。在原六小时线性MPC网络与资源约束下，求SCS/FCS/ESS/PCC下一小时功率的支持极值，并求下一小时恢复至同future时点参考ESS余量所需的最小缺口rho。业务目标不能超过该候选状态自身最优值的5%增量。rho为零意味着该预算内能恢复；正缺口表示恢复受限，不直接等同于电网故障。5%及一小时期限是审查任务设置，未改变四类风险的原始判据。

![全状态全剂量恢复图谱](recovery_atlas.png)

图谱显示完整六个source、两族动作和所有14个剂量（含公共端点）。颜色是连续储备缺口，黑线为相邻格的恢复状态转换。它解释边界在哪里、是否随接续时间移动，以及不同source是否具有不同可恢复区域。所计算的支持区间属于给定业务预算内的线性MPC计划集合，不是完整AC域体积。

![不同动作类别的调度范围变化](dispatch_shapes.png)

这张组合图把所有内点动作纳入类别统计。上排展示每个source、每类动作在三个future时点、四类调度方向的中位变化；下排给出全剂量ESS响应的source中位数与四分位范围。变化以固定名义功率标尺归一化，正值表示扩大、负值表示收缩；因此可以检验资源释放是否在后续某些方向上留下收缩，以及扩大/收缩是否随时间转向，而无需只挑一个代表动作。

## 方法贡献如何分开判断

Random与Sobol在同池采样；Error critics只学习冻结旧模型对18个同名物理端点的配对预测误差；Boundary reward only只改变学习目标；Bracket + random只使用已揭示可恢复/不可恢复端点细化一半查询；Boundary critics + pairing把新目标与同样的成对细化结合。后三者用于判断提升来自学习到的画像价值，还是来自显式的括区协议，不能将二分细化的全部增益归给RL。

新奖励把固定标尺的画像新颖性、已揭示相邻动作的恢复差异、范围扩大/收缩的转折差异各占三分之一。Q训练只见已经付费揭示的经验；未来视图、缺口和全集边界不在公共候选特征内。各学习臂统一接收18维已查询知识摘要，描述附近动作、已知两侧恢复状态和括区；新学习仍使用原128宽双critic，未增加网络层数或宽度。采集为gamma=0的有限池上下文任务，actor不作为本轮提案器。

## source级统计

先在各source内平均三个seed，再对独立source作配对bootstrap 2000次。以下为混合采集器在64查询时相邻边界定位率的绝对增益：

'''
    for base,value in ci.items():text+=f"相对{names[base]}：{value['gain']*100:+.2f}pp，95% CI [{value['ci95'][0]*100:+.2f}, {value['ci95'][1]*100:+.2f}]，有效source={value['independent_sources']}。\n\n"
    text+=f'''## 物理核验与计算成本

共同库共{support['total_branches']}分支，{support['known_physical_hours']}小时AC核验通过，未核验{support['unverified_hours']}小时，未知完整画像{support['unknown_descriptors']}条。逐小时检查电压0.93–1.07pu、0.9kA线路限额、机组边界、节点平衡和车辆/储能能量守恒。首动作发生有界裁剪{clipped}条、预测分配保护调整{guards}条。额外有功处置{extra}小时，记录非EV削负荷{shed:.3f}MWh；这些是当前保护控制器输出，未被解释成物理上不可避免的最低代价。

共同建库的真实仿真费用与每臂答案揭示预算分开：162条分支中匹配复用已有6条，新增156条；每臂128答案按8/16/32/64/128前缀比较。该轮检验六个历史初态上的采集与细化能力，不把全库预计算成本隐藏为在线节约。预警器和旧模型训练调用均为0；新采集模型三种学习配置×3seed共9次CPU拟合。

## 执行与产物

项目：D:\\博士资料包\\科研\\科研\\AAAI_RL\\tarp_37_information_acquisition。

```powershell
Set-Location -LiteralPath 'D:\\博士资料包\\科研\\科研\\AAAI_RL\\tarp_37_information_acquisition'
powershell -NoProfile -ExecutionPolicy Bypass -File .\\scripts\\run_boundary_acquisition.ps1 -RunId {args.run_id} -Stage fit
```

Stage fit验证并复用已有教师、公共/私有数据和全部采集结果，仅重建分析。新完整实验使用新RunId与Stage all。原始分支、MPC求解状态、边界对账本、各seed查询顺序、输入与模型hash均在artifacts/boundary_v2/{args.run_id}。
'''
    (out/'results.md').write_text(text,encoding='utf-8')
    import markdown
    body=markdown.markdown(text,extensions=['tables','fenced_code'])
    css='body{max-width:1180px;margin:35px auto;padding:0 25px;font:17px/1.85 "Microsoft YaHei",sans-serif;color:#263a49}h1{font-size:29px}h2{font-size:23px;margin-top:36px}img{width:100%}table{width:100%;border-collapse:collapse;font-size:15px}td,th{border:1px solid #d9e3e8;padding:9px}th{background:#edf3f6}pre{padding:16px;background:#f4f7f8;overflow:auto}code{font-size:14px}'
    (out/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>后续调度边界采集</title><style>'+css+'</style><body>'+body+'</body></html>',encoding='utf-8')
    print(json.dumps(dict(complete=True,primary_budget=64,support=support,table=[x for x in table if x['budget']==64],ci=ci),ensure_ascii=False))


def save(fig,out,name,plt):
    fig.savefig(out/(name+'.png'),dpi=220);fig.savefig(out/(name+'.svg'));plt.close(fig)
    path=out/(name+'.svg');path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')


if __name__=='__main__':main()
