from numpy.linalg import norm

def re_max(input_tensor, dim):

    x = F.relu(input_tensor)
    sum = torch.sum(x, dim)
    x = x / sum              # To do: check if it actually does what I want it to do
    return x                 # i.e. check if it sums and divides for the right numbers
