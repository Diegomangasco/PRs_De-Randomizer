from torch.utils.tensorboard import SummaryWriter
from torch import rand, float32
from model import ProbesEncoder

writer = SummaryWriter("torchlogs/")
model = ProbesEncoder(305, 150, 50)
x = rand(32, 305, dtype=float32)
writer.add_graph(model, x)
writer.close()