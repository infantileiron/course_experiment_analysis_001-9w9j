"""User-level fixed-horizon randomized experiment analysis."""
from datetime import datetime, timezone
import math
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import NormalIndPower

METRICS = {'completion_rate': '结课率', 'study_minutes': '人均学习分钟', 'bounce_rate': '未学习率'}

def demo_events(scenario='uplift', seed=41, n=1600):
    rng=np.random.default_rng(seed)
    rows=[]
    for i in range(n):
        group='B' if rng.random() < (.72 if scenario=='ratio_anomaly' else .5) else 'A'
        segment='new' if rng.random()<.55 else 'returning'
        t=pd.Timestamp('2026-01-05',tz='UTC')+pd.Timedelta(hours=int(rng.integers(0,300)))
        base=.42+(.13 if scenario=='uplift' and group=='B' else 0)
        minute=max(0,float(rng.normal(34+(7 if scenario=='uplift' and group=='B' else 0),16)))
        rows.append({'user_id':str(i),'group':group,'event':'exposure','time':t.isoformat(),'minutes':0.,'segment':segment})
        rows.append({'user_id':str(i),'group':group,'event':'exposure','time':(t+pd.Timedelta(days=1)).isoformat(),'minutes':0.,'segment':segment})
        if rng.random()>.08:
            rows.append({'user_id':str(i),'group':group,'event':'study','time':(t+pd.Timedelta(hours=1)).isoformat(),'minutes':round(minute,3),'segment':segment})
        if rng.random()<base:
            rows.append({'user_id':str(i),'group':group,'event':'complete','time':(t+pd.Timedelta(hours=2)).isoformat(),'minutes':0.,'segment':segment})
    return rows

def validate_events(rows):
    if not rows or len(rows)>500000: raise ValueError('数据不能为空，最多支持50万行')
    frame=pd.DataFrame(rows)
    required={'user_id','group','event','time','minutes','segment'}
    if not required.issubset(frame.columns): raise ValueError('缺少字段：'+','.join(sorted(required-set(frame.columns))))
    if frame[list(required)].isnull().any().any(): raise ValueError('必填字段不能有空值')
    if not set(frame['group']).issubset({'A','B'}): raise ValueError('group只能是A或B')
    if not set(frame.event).issubset({'exposure','study','complete'}): raise ValueError('event只能是exposure、study、complete')
    frame['minutes']=pd.to_numeric(frame.minutes,errors='coerce')
    if not np.isfinite(frame.minutes).all() or (frame.minutes<0).any(): raise ValueError('minutes必须是有限非负数')
    frame['time']=pd.to_datetime(frame.time,utc=True,errors='coerce',format='mixed')
    if frame.time.isnull().any(): raise ValueError('time必须是有效时间戳')
    frame['user_id']=frame.user_id.astype(str)
    if frame.user_id.str.strip().eq('').any(): raise ValueError('user_id不能为空')
    return frame

def validate_config(config):
    out={k:config.get(k) for k in ['name','dataset_id','start','end','primary_metric']}
    if not out['name'] or not out['dataset_id']: raise ValueError('名称和数据版本必填')
    if out['primary_metric'] not in METRICS: raise ValueError('无效主要指标')
    start,end=pd.to_datetime(out['start'],utc=True),pd.to_datetime(out['end'],utc=True)
    if end<=start: raise ValueError('结束时间必须晚于开始时间')
    out.update(start=start.isoformat(),end=end.isoformat(),alpha=float(config.get('alpha',.05)),mde=float(config.get('mde',.05)),allocation_b=float(config.get('allocation_b',.5)),planned_segments=list(config.get('planned_segments',[])))
    if not 0<out['alpha']<.2 or not 0<out['mde'] or not .05<=out['allocation_b']<=.95: raise ValueError('alpha须在0至0.2，MDE须为正数，B组比例须在0.05至0.95之间')
    if out['primary_metric']!='study_minutes' and out['mde']>=1: raise ValueError('比率指标MDE须小于1')
    return out

def _comparison(a,b,binary,alpha):
    a,b=np.asarray(a,float),np.asarray(b,float)
    if len(a)<2 or len(b)<2:return {'status':'insufficient','n_a':len(a),'n_b':len(b),'reason':'每组至少需要2名用户'}
    ma,mb=float(a.mean()),float(b.mean());diff=mb-ma
    if binary:
        se=math.sqrt(ma*(1-ma)/len(a)+mb*(1-mb)/len(b))
        pool=(a.sum()+b.sum())/(len(a)+len(b))
        nullse=math.sqrt(pool*(1-pool)*(1/len(a)+1/len(b)))
        p=float(2*stats.norm.sf(abs(diff/nullse))) if nullse>0 else 1.
        crit=float(stats.norm.ppf(1-alpha/2));method='两独立比例正态近似检验'
        note='置信区间为未校正的比例差正态近似区间；小样本或极端比例时近似不稳定'
    else:
        va,vb=float(a.var(ddof=1)),float(b.var(ddof=1));se=math.sqrt(va/len(a)+vb/len(b))
        denom=(va/len(a))**2/(len(a)-1)+(vb/len(b))**2/(len(b)-1)
        df=(va/len(a)+vb/len(b))**2/denom if denom>0 else len(a)+len(b)-2
        crit=float(stats.t.ppf(1-alpha/2,df));p=float(2*stats.t.sf(abs(diff/se),df)) if se>0 else (1. if diff==0 else 0.)
        method='Welch独立样本t检验';note='置信区间未作多重比较校正'
    return {'status':'ok','n_a':len(a),'n_b':len(b),'mean_a':ma,'mean_b':mb,'effect':diff,'relative_effect':diff/ma if ma else None,'ci_low':diff-crit*se,'ci_high':diff+crit*se,'p_value':p,'method':method,'note':note}

def analyze(rows,config,now=None,explore_segment=None):
    config=validate_config(config);frame=validate_events(rows)
    start,end=pd.to_datetime(config['start'],utc=True),pd.to_datetime(config['end'],utc=True)
    frame=frame[(frame.time>=start)&(frame.time<end)].copy()
    exposures=frame[frame.event=='exposure'].sort_values('time')
    conflicts=exposures.groupby('user_id')['group'].nunique();bad=set(conflicts[conflicts>1].index)
    exposure_segments=exposures.groupby('user_id')['segment'].nunique();segment_conflicts=set(exposure_segments[exposure_segments>1].index)
    roster=exposures[~exposures.user_id.isin(bad)].drop_duplicates('user_id').copy()
    if roster.empty: raise ValueError('窗口内没有可分析的曝光用户')
    valid=frame.merge(roster[['user_id','group','time']].rename(columns={'time':'first_exposure'}),on=['user_id','group'])
    valid=valid[valid.time>=valid.first_exposure]
    complete=set(valid.loc[valid.event=='complete','user_id'])
    minutes=valid.loc[valid.event=='study'].groupby('user_id').minutes.sum()
    roster['completion_rate']=roster.user_id.isin(complete).astype(float)
    roster['study_minutes']=roster.user_id.map(minutes).fillna(0)
    roster['bounce_rate']=(roster.study_minutes==0).astype(float)
    counts=roster.group.value_counts();na,nb=int(counts.get('A',0)),int(counts.get('B',0));n=na+nb
    expected=[n*(1-config['allocation_b']),n*config['allocation_b']]
    srm=float(stats.chisquare([na,nb],expected).pvalue)
    metrics=[]
    for metric,label in METRICS.items():
        result=_comparison(roster.loc[roster.group=='A',metric],roster.loc[roster.group=='B',metric],metric!='study_minutes',config['alpha'])
        result.update(metric=metric,label=label,role='主要指标' if metric==config['primary_metric'] else '护栏指标')
        metrics.append(result)
    valid_metrics=[m for m in metrics if m['status']=='ok']
    if valid_metrics:
        for m,p,reject in zip(valid_metrics,multipletests([m['p_value'] for m in valid_metrics],alpha=config['alpha'],method='holm')[1],multipletests([m['p_value'] for m in valid_metrics],alpha=config['alpha'],method='holm')[0]):
            m.update(adjusted_p=float(p),significant=bool(reject))
    primary=next(m for m in metrics if m['metric']==config['primary_metric'])
    mde=config['mde'];needed=None
    if na>1:
        baseline=float(roster.loc[roster.group=='A',config['primary_metric']].mean())
        if config['primary_metric']=='study_minutes':
            sd=float(roster.loc[roster.group=='A','study_minutes'].std())
            effect=mde/sd if sd>0 else None
        else:
            alternate=min(.999999,max(.000001,baseline+mde))
            effect=2*abs(math.asin(math.sqrt(alternate))-math.asin(math.sqrt(baseline))) if baseline+mde<1 else None
        if effect and effect>0:
            required=NormalIndPower().solve_power(effect_size=effect,alpha=config['alpha']/3,power=.8,ratio=config['allocation_b']/(1-config['allocation_b']))
            needed=int(math.ceil(required))
    segments=[]
    selected=list(dict.fromkeys(config['planned_segments']+([explore_segment] if explore_segment else [])))
    for segment in selected:
        sub=roster[(roster.segment==segment)&(~roster.user_id.isin(segment_conflicts))]
        result=_comparison(sub.loc[sub.group=='A',config['primary_metric']],sub.loc[sub.group=='B',config['primary_metric']],config['primary_metric']!='study_minutes',config['alpha'])
        result.update(segment=segment,kind='预先约定' if segment in config['planned_segments'] else '事后探索（不能作确认性结论）')
        segments.append(result)
    segment_ok=[x for x in segments if x['status']=='ok']
    if segment_ok:
        for item,p in zip(segment_ok,multipletests([x['p_value'] for x in segment_ok],method='holm')[1]):item['adjusted_p']=float(p)
    current=pd.Timestamp(now or datetime.now(timezone.utc))
    if current.tzinfo is None: current=current.tz_localize('UTC')
    warnings=[]
    finished=current>=end
    if not finished:warnings.append('实验尚未结束：当前结果仅用于监测，不支持据此提前停止或作最终效果声明。')
    if srm<.001:warnings.append('分组比例异常：应先排查分流、曝光埋点和样本丢失，当前效果不宜直接用于决策。')
    if bad:warnings.append(f'{len(bad)}名用户跨组曝光，已从两组分析中排除；排除可能影响随机性。')
    if segment_conflicts:warnings.append(f'{len(segment_conflicts)}名用户的人群标签发生变化，分层分析中排除，总体保留。')
    warnings.append('采用固定结束时间分析；用户是独立分析单位。相同用户多次曝光只计一次，学习时长累加。')
    if min(na,nb)<30:warnings.append('组内样本较少，正态近似结论可能不稳定。')
    if primary.get('significant') and finished and srm>=.001:
        direction='提高' if primary['effect']>0 else '降低'
        conclusion=f"主要指标{METRICS[config['primary_metric']]}在Holm校正后存在统计差异，B组{direction}；同时检查护栏和实际业务幅度。"
    else:conclusion='当前证据不足以形成已验证的上线收益结论；请结合实验完成状态、分流诊断、置信区间及护栏指标。'
    return {'counts':{'A':na,'B':nb},'raw_exposure_events':len(exposures),'excluded_crossovers':len(bad),'segment_conflicts':len(segment_conflicts),'srm_p_value':srm,'metrics':metrics,'segments':segments,'finished':finished,'warnings':warnings,'conclusion':conclusion,'sample_plan':{'mde':mde,'power':.8,'required_a_approx':needed,'required_b_approx':int(math.ceil(needed*config['allocation_b']/(1-config['allocation_b']))) if needed else None,'note':'基于当前A组估计的近似规划；三指标Bonferroni保守规划，不能作为事后功效证明。'},'distribution':{g:{'minutes_histogram':np.histogram(roster.loc[roster.group==g,'study_minutes'],bins=[0,10,20,30,40,50,60,90,10000])[0].tolist(),'bins':['0–10','10–20','20–30','30–40','40–50','50–60','60–90','90+']} for g in ['A','B']}}
