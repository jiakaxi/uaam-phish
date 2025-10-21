import torch
from types import SimpleNamespace
from src.systems.url_only_module import UrlOnlySystem

class DummyModel(torch.nn.Module):
    def __init__(self): super().__init__(); self.fc = torch.nn.Linear(4,1)
    def forward(self, batch): return self.fc(batch["x"]).view(-1)

def test_url_only_system_step():
    cfg = SimpleNamespace(train=SimpleNamespace(lr=1e-3, weight_decay=0.0))
    model = DummyModel()
    sys = UrlOnlySystem(cfg, model)
    batch = {"x": torch.randn(2,4), "label": torch.tensor([0.0,1.0])}
    loss = sys.step(batch, "train")
    assert torch.isfinite(loss)
