

# BEFORE execute this script,
# load libraries, functions/classes and various settings
# by vit.py script.


###################################################################################################
# -------------------------------------------------------------------------------------------------
# check step by step:  Vision Transformer (ViT)
# -------------------------------------------------------------------------------------------------

SEED = 3497
reproducibility(SEED)

image_size = 32

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(image_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

print(transform_train)


# ----------
batch_size = 1

trainset = torchvision.datasets.CIFAR10(root='./01_data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./01_data', train=False, download=True, transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
num_classes = len(classes)


# ----------
N = 2
M = 14
transform_train.transforms.insert(0, RandAugment(N, M))

# Note that Rand Augment is inserted at first operation
print(transform_train)


# ----------
# VIT initialization
image_size = 32
patch_size = 4
channels = 3

image_height, image_width = image_size, image_size
patch_height, patch_width = patch_size, patch_size

num_patches = (image_height // patch_height) * (image_width // patch_width)
# 64
print(num_patches)

channels = 3
patch_dim = channels * patch_height * patch_width
# patch_dim = 3 * 4 * 4 = 48
print(patch_dim)

dim = 512
to_patch_embedding = nn.Sequential(
    # b: batch size  c: channel  h: 8  p1: 4  w: 8  p2: 4
    Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
    nn.LayerNorm(patch_dim),
    nn.Linear(patch_dim, dim),
    nn.LayerNorm(dim),
)


pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
print(pos_embedding.shape)

# cls_token is random value from normal distribution (mean 0, std 1)
cls_token = nn.Parameter(torch.randn(1, 1, dim))
print(cls_token.shape)

# ----------
# get 1 batch
inputs, targets = next(iter(trainloader))

device = 'cuda'
inputs, targets = inputs.to(device), targets.to(device)


# ----------
# Linear Projection of Flattened Patches

x = to_patch_embedding.to('cuda')(inputs)
print(inputs.shape)
# (1, num_patches(64), dimhead(512))
print(x.shape)
b, n, _ = x.shape


# ----------
# repeat batch size
cls_tokens = repeat(cls_token, '1 1 d -> b 1 d', b=b)
print(cls_tokens.shape)


# ----------
# Patch + Position Embedding
x = torch.cat((cls_tokens.to('cuda'), x), dim=1)
# now cls_tokens is concat to 1st axis (1st axis dim 64 --> 64+1)
print(x.shape)
x += pos_embedding.to('cuda')[:, :(n + 1)]
x = dropout(x)
print(x.shape)


# ------------------------
# Transformer Encoder
# ------------------------

# depth 1:  Attention

dim = 512
x2 = nn.LayerNorm(dim).to('cuda')(x)
print(x2.shape)

dim_head = 64
heads = 8
tmp = nn.Linear(dim, dim_head * heads * 3, bias=False).to('cuda')(x2)
qkv = tmp.chunk(3, dim=-1)

# dim_head * heads * 3 = 1536
print(tmp.shape)
print(len(qkv))
print(qkv[0].shape)
print(qkv[1].shape)
print(qkv[2].shape)

# d is dim_head = 64
# n: 64 + 1 = 65
q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=heads), qkv)
print(q.shape)
print(k.shape)
print(v.shape)

scale = dim_head ** -0.5
# transpose n d --> d n
dots = torch.matmul(q, k.transpose(-1, -2)) * scale
# (batch_size, heads, n, n)  (n = 64 + 1 = num_patches + 1)
print(dots.shape)
print(k.transpose(-1, -2).shape)

attn = nn.Softmax(dim=-1)(dots)
attn = nn.Dropout(0.1)(attn)
print(attn.shape)

out = torch.matmul(attn, v)

# all heads again into dim (dim = h * d)
out = rearrange(out, 'b h n d -> b n (h d)')
# (batch_size, n, dim)
print(out.shape)

to_out = nn.Identity()(out)
print(to_out.shape)


# depth 1:  Attention + x
x3 = to_out + x
print(x3.shape)


# depth 1:  Feedforward
mlp_dim = 512
ff_net = nn.Sequential(
    nn.LayerNorm(dim),
    nn.Linear(dim, mlp_dim),
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(mlp_dim, dim),
    nn.Dropout(0.1)
)

x4 = ff_net.to('cuda')(x3) + x3
# (batch_size, n, dim)
print(x4.shape)


# --> depth 2:  Attention ...  (continue to depth 6)

# final after depth 6
final_x = nn.LayerNorm(dim).to('cuda')(x4)
print(final_x.shape)


# ------------------------
# MLP Head
# ------------------------

final_x2 = final_x[:, 0]
print(final_x2.shape)

# MLP head:  linear projection dim (512) to num_classes (10)
final_mlp_head = nn.Linear(dim, num_classes).to('cuda')(final_x2)
print(final_mlp_head.shape)

