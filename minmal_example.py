import psutil
import torch.nn


class Linear(torch.nn.Module):
    def __init__(self):
        super(Linear, self).__init__()
        self.a = torch.nn.Linear(1, 3)
        self.b = torch.nn.Linear(1, 3)
        self.b.load_state_dict(self.a.state_dict())
        for p in self.b.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.a(x)


l = Linear()
l.to('cuda')
opt = torch.optim.Adam(l.a.parameters(), lr=0.01)
while True:
    x = torch.randn(1, 1).to('cuda')
    y = l(x)
    y.sum().backward()
    opt.step()
    opt.zero_grad()
    with torch.no_grad():
        for la, lb in zip(l.a.parameters(), l.b.parameters()):
            lb.data.mul_(0.9)
            lb.data.add_(0.1 * la)
    print(psutil.virtual_memory().percent)
