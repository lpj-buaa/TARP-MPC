import torch
from tarp_boundary.learning import acquire
from tarp_info.cache import Oracle


def test_primary_budget_checkpoint_is_the_actual_model_at_64_queries(tmp_path):
    initial=[dict(id='ref',key='a',family='reference',dose=0.,state=[0.],action=[0.]*51,
        descriptor=[0.],recoverable=[True]*3,turnover=[False]*4,error=0.)]
    public=[dict(id=str(j),key='a',family='release',dose=j/81,state=[0.],action=[j/81]*51) for j in range(1,81)]
    answers={p['id']:dict(descriptor=[p['dose']],recoverable=[p['dose']<.5]*3,turnover=[False]*4,error=p['dose']) for p in public}
    path=tmp_path/'model.pt';acquire(public,initial,Oracle(answers),'boundary_td3',64,42,path)
    saved=torch.load(tmp_path/'model_b64.pt',map_location='cpu',weights_only=False)
    final=torch.load(path,map_location='cpu',weights_only=False)
    assert saved['state_dim']==19
    assert all(torch.equal(saved['q1'][k],v) for k,v in final['q1'].items())
