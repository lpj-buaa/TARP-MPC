import numpy as np
import torch


def seeded_agent():
    from tarp37.acquisition import AcquisitionTD3
    agent=AcquisitionTD3(3,'cpu',42)
    for i in range(5): agent.add([float(i),0.,1.],np.zeros(51),float(i))
    return agent


def test_random_and_sobol_controls_do_not_fit_an_unused_rl_policy():
    from tarp37.acquisition import warm_start_acquisition
    for policy in ('random','sobol'):
        agent=seeded_agent(); before=[p.detach().clone() for p in agent.actor.parameters()]
        warm_start_acquisition(agent,policy,steps=4)
        assert agent.steps==0
        assert all(torch.equal(p,q) for p,q in zip(agent.actor.parameters(),before))


def test_td3_information_sampler_still_warms_up_on_existing_training_queries():
    from tarp37.acquisition import warm_start_acquisition
    agent=seeded_agent()
    warm_start_acquisition(agent,'td3',steps=4)
    assert agent.steps==4
