import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()
        self.transformer = SelfAttention(256)

        self.conv1 = nn.Conv2d(256, 1, kernel_size=1)
        self.conv2 = nn.Conv2d(512, 1, kernel_size=1)
        self.conv3 = nn.Conv2d(1024, 1, kernel_size=1)

        self.DoubleConv = DoubleConv(1536)

    def forward(self, x1, x2, x3):
        b, c, h, w = x3.shape
        p1 = F.interpolate(x1, h, mode="bilinear")
        p2 = F.interpolate(x2, h, mode="bilinear")

        p2 = self.conv2(p2).view(b, 1, 256).permute(0, 2, 1)
        x3 = self.conv3(x3).view(b, 1, 256).permute(0, 2, 1)
        p1 = self.conv1(p1).view(b, 1, 256).permute(0, 2, 1)

        list1 = []
        list2 = []

        list1.append(p1)
        list1.append(p2)
        list1.append(x3)

        for i in range(len(list1)):
            if i == 0:
                atten = self.transformer(list1[i], list1[i])
                atten2 = self.transformer(list1[i], list1[i + 1])
                atten3 = self.transformer(list1[i], list1[i + 2])
                list2.append(atten)
                list2.append(atten2)
                list2.append(atten3)
            elif i == 1:
                atten = self.transformer(list1[i], list1[i])
                atten2 = self.transformer(list1[i], list1[i + 1])
                list2.append(atten)
                list2.append(atten2)
            else:
                atten = self.transformer(list1[i], list1[i])
                list2.append(atten)

        out = torch.concat(list2, dim=1)
        out = self.DoubleConv(out)
        return out

class DoubleConv(nn.Sequential):
    def __init__(self, in_channels):
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.GELU()
        )


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, n_heads=8):
        super(SelfAttention, self).__init__()
        self.hid_dim = hidden_size
        self.n_heads = n_heads

        assert hidden_size % n_heads == 0
        self.w_q = nn.Linear(1, hidden_size)
        self.w_k = nn.Linear(1, hidden_size)
        self.w_v = nn.Linear(1, hidden_size)
        self.fc = nn.Linear(hidden_size, hidden_size)

    def sinusoidal_position_embedding(self, batch_size, nums_head, max_len, output_dim, device):
        # (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(-1)
        # (output_dim//2)
        ids = torch.arange(0, output_dim // 2, dtype=torch.float)  
        theta = torch.pow(10000, -2 * ids / output_dim)

        # (max_len, output_dim//2)
        embeddings = position * theta  

        # (max_len, output_dim//2, 2)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

        # (bs, head, max_len, output_dim//2, 2)
        embeddings = embeddings.repeat((batch_size, nums_head, *([1] * len(embeddings.shape))))  

        # (bs, head, max_len, output_dim)
        embeddings = torch.reshape(embeddings, (batch_size, nums_head, max_len, output_dim))
        embeddings = embeddings.to(device)

        return embeddings

    def RoPE(self, q, k):
        # q,k: (bs, head, max_len, output_dim)
        q = q.unsqueeze(1)
        k = k.unsqueeze(1)
        batch_size = q.shape[0]
        nums_head = q.shape[1]
        max_len = q.shape[2]
        output_dim = q.shape[-1]

        # (bs, head, max_len, output_dim)
        pos_emb = self.sinusoidal_position_embedding(batch_size, nums_head, max_len, output_dim, q.device)

        # cos_pos,sin_pos: (bs, head, max_len, output_dim)
        cos_pos = pos_emb[..., 1::2].repeat_interleave(2, dim=-1)
        sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)

        # q,k: (bs, head, max_len, output_dim)
        q2 = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1)
        q2 = q2.reshape(q.shape)  

        q = q * cos_pos + q2 * sin_pos

        k2 = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1)
        k2 = k2.reshape(k.shape)
 
        k = k * cos_pos + k2 * sin_pos

        return q.squeeze(1), k.squeeze(1)

    def forward(self, text_embedding, embedding, mask=None):
        b = text_embedding.shape[0]
        bsz = text_embedding.shape[0]
        Q = self.w_q(text_embedding)
        K = self.w_k(embedding)

        Q, K = self.RoPE(Q, K)

        V = self.w_v(embedding)


        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)

        scaling = K.shape[2]
        attention = torch.matmul(Q, K.permute(0, 1, 3, 2)) / math.sqrt(scaling)

        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(attention, dim=-1)

        x = torch.matmul(attention, V)

        x = x.permute(0, 2, 1, 3).contiguous()

        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))

        x = self.fc(x)

        return x.view(b, 256, 16, 16)


if __name__ == "__main__":
    net = Fusion(2048, 2048)
    x1 = torch.randn(1, 256, 64, 64)
    x2 = torch.randn(1, 512, 32, 32)
    x3 = torch.randn(1, 1024, 16, 16)
    y = net(x1, x2, x3)

