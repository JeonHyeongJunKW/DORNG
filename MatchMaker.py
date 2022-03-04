import torch
import torch.nn.functional as F


def find_local_matches(desc1, desc2, kernel_size=9):
    # Computes the correlation between each pixel on desc1 with all neighbors
    # inside a window of size (kernel_size, kernel_size) on desc2. The match
    # vector if then computed by linking each pixel on desc1 with
    # the pixel with desc2 with the highest correlation.
    #
    # This approch requires a lot of memory to build the unfolded descriptor.
    # A better approach is to use the Correlation package from e.g.
    # https://github.com/NVIDIA/flownet2-pytorch/tree/master/networks/correlation_package
    desc2_unfolded = F.unfold(desc2, kernel_size, padding=kernel_size//2)
    desc2_unfolded = desc2_unfolded.reshape(
        1, desc2.shape[1], kernel_size*kernel_size, desc2.shape[2], desc2.shape[3])
    desc1 = desc1.unsqueeze(dim=2)
    correlation = torch.sum(desc1 * desc2_unfolded, dim=1)
    _, match_idx = torch.max(correlation, dim=1)
    hmatch = torch.fmod(match_idx, kernel_size) - kernel_size // 2#height가 아니라 horizontal이다.
    vmatch = match_idx // kernel_size - kernel_size // 2#vertical이다.
    matches = torch.cat((hmatch, vmatch), dim=0)
    return matches