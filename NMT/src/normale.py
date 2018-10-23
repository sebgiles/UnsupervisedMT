from numpy.linalg import norm


encoded_1 = trainer.encoder(sent1, len1, lang1_id)
encoded_2 = trainer.encoder(sent2, len2, lang2_id)

loss += norm_loss(encoded_1, encoded_2, dist_sample)


def re_max(input_tensor, dim):

    x = F.relu(input_tensor)
    sum = torch.sum(x, dim)
    x = x / sum              # To do: check if it actually does what I want it to do
    return x                 # i.e. check if it sums and divides for the right numbers

def norm_loss(points, targets, dist_sample):

    poin_targ = sum(norm(points - targets))
    sum_dis = sum(norm(dist_sample))
    n_loss = poin_targ / sum_dis
    return n_loss


def check_avg_dist(dist_sample):

    avg_dis = sum(norm(dist_sample)) / len(dist_sample)
    return avg_dis






self.zero_grad(['enc', 'dec'])
loss.backward()
self.update_params(['enc', 'dec'])



latent = encoder_out['encoder_out'] #line 273 transformer. seems useful
