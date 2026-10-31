"""Matched direct and GRU decoders; future supervision never enters forward."""
import torch
from torch import nn
from torch.nn import functional as F


class ConsequenceStudent(nn.Module):
    def __init__(self, schema, decoder='direct', width=64):
        super().__init__()
        if decoder not in ('direct', 'gru') or width % 4:
            raise ValueError('Decoder direct/gru and width divisible by4 required')
        self.decoder = decoder
        shapes = schema['input_shapes']; self.outcome_dim = len(schema['outcome_fields'])
        self.register_buffer('adjacency', torch.tensor(schema['adjacency'], dtype=torch.float32))
        self.node_in = nn.Linear(shapes['nodes'][-1], width)
        self.graph_layers = nn.ModuleList([nn.Linear(width, width) for _ in range(2)])
        self.history_in = nn.Linear(shapes['history'][-1], width)
        self.position = nn.Parameter(torch.randn(10, width) * .02)
        layer = nn.TransformerEncoderLayer(width, 4, width * 2, dropout=0., batch_first=True)
        self.history_encoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.fuse = nn.Sequential(nn.Linear(width * 3 + shapes['context'][0] + 51 + 24, width),
                                  nn.GELU(), nn.LayerNorm(width))
        self.step = nn.Embedding(6, width)
        if decoder == 'direct':
            # Match the transition parameter budget of GRUCell(input=width+116, hidden=width).
            expansion = round((6 * width * width + 353 * width) / (3 * width + 5))
            self.transition = nn.Sequential(nn.Linear(width * 2 + 4, expansion), nn.GELU(),
                                             nn.Linear(expansion, width), nn.GELU())
        else:
            self.transition = nn.GRUCell(width + 4 + 6 + 51 + 4 + 51, width)
        self.readout = nn.Linear(width, self.outcome_dim + 6 + 51 + 4)
        self.event_head = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), nn.Linear(width, 4))

    def forward(self, batch):
        nodes = F.gelu(self.node_in(batch['nodes'].float()))
        for layer in self.graph_layers:
            nodes = nodes + F.gelu(layer(torch.matmul(self.adjacency, nodes)))
        history = self.history_in(batch['history'].float()) + self.position
        history = self.history_encoder(history).mean(1)
        z = self.fuse(torch.cat([nodes.mean(1), nodes.amax(1), history,
            batch['context'].float(), batch['action'].float(), batch['forecast'].float().flatten(1)], -1))
        resource, response, weight = (batch[k].float() for k in ['initial_resource', 'initial_response', 'initial_weight'])
        hidden = z; outputs, states = [], []
        for h in range(6):
            step = self.step.weight[h].expand(len(z), -1)
            forecast = batch['forecast'][:, h].float()
            if self.decoder == 'direct':
                hidden = self.transition(torch.cat([z, step, forecast], -1))
            else:
                intervention = batch['candidate_request'].float() if h == 0 else torch.zeros_like(response)
                hidden = self.transition(torch.cat([step, forecast, resource, response, weight, intervention], -1), hidden)
            raw = self.readout(hidden)
            outcome, r, u, w = torch.split(raw, [self.outcome_dim, 6, 51, 4], -1)
            resource = torch.sigmoid(r)
            # A site allocation is not its realized EV delivery. Even at h=0,
            # arrivals, availability and protection determine actual response.
            response = u
            weight = batch['initial_weight'].float() if h == 0 else torch.softmax(w, -1)
            outputs.append((outcome, resource, response, weight)); states.append(hidden)
        result = {name: torch.stack([x[j] for x in outputs], 1)
                  for j, name in enumerate(['outcome', 'resource', 'response', 'weight'])}
        result['event'] = self.event_head(torch.cat([z, torch.stack(states, 1).mean(1)], -1))
        return result


def masked_huber(predicted, target, mask):
    mask = mask.bool()
    return F.smooth_l1_loss(predicted[mask], target.float()[mask]) if mask.any() else predicted.sum() * 0.


def loss_terms(predicted, batch, structured=True, positive_weight=None):
    terms = {'outcome': masked_huber(predicted['outcome'], batch['outcome'], batch['outcome_mask'])}
    event_mask = batch['event_mask'].bool()
    event_loss = F.binary_cross_entropy_with_logits(predicted['event'], batch['event'].float(),
                                                   pos_weight=positive_weight, reduction='none')
    terms['event'] = event_loss[event_mask].mean() if event_mask.any() else predicted['event'].sum() * 0.
    refs = batch['reference'].long()
    mask = batch['outcome_mask'].bool() & batch['outcome_mask'][refs].bool()
    mask = mask & (refs != torch.arange(len(refs), device=refs.device))[:, None, None]
    terms['delta'] = masked_huber(predicted['outcome'] - predicted['outcome'][refs],
                                  batch['outcome'] - batch['outcome'][refs], mask)
    for name in ['resource', 'response', 'weight']:
        terms[name] = masked_huber(predicted[name], batch[name], batch[name + '_mask'])
    terms['total'] = terms['outcome'] + terms['event']
    if structured:
        terms['total'] = terms['total'] + sum(terms[k] for k in ['delta', 'resource', 'response', 'weight'])
    return terms
