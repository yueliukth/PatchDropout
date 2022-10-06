import torch

class PatchDropout(torch.nn.Module):
    """ 
    Implements PatchDropout: https://arxiv.org/abs/2208.07220
    """
    def __init__(self, keep_rate=0.5, sampling="random"):
        super().__init__()
        assert 0 < keep_rate <=1, "The keep_rate must be in (0,1]"
        
        self.keep_rate = keep_rate
        self.sampling = sampling

    def forward(self, x):
        if self.keep_rate == 1: return x
        # generating patch mask
        x = self.get_mask(x)
        return x

    def get_mask(self, x):
        if self.sampling == "random":
            return self.random_mask(x)
        else:
            return NotImplementedError(f"PatchDropout does ot support {self.sampling} sampling")
    
    def random_mask(self, x):
        """
        Returns an id-mask using uniform sampling
        """
        N, L, D = x.shape  # batch, length, dim
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        len_keep = int(L * self.keep_rate)

        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove

        # Keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]

        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        return x_masked

