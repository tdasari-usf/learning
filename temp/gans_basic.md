

Convention - Discriminator predicts real images as label 1, fake as label 0

Discriminator Loss function 

For the discriminator to be great it should be able to detect fake as fake, real as real. Which leads to log(~1) + log(~1), which upper bounds
the loss function to 0, and lower bounds to negative infinity.
L = log(D(real)) + log(1 - D(G(z)))

so we maximize L.

Convention of NN is to minimize the loss, to follow that we use nn.BCELoss - which has a negative sign, hence we minimize loss.


Generator Loss function - For the generator to be great the discriminator should not be able to detecte the fakes as fakes, instead should
predict the fake ones as real, implies D(G(z)) ~ 1, which makes the below Loss function to log(~0) i.e negative infinity, which is why
we would like to minimize the below L.

L1 = log(1 - D(G(z)))

To get to stable gradients (as mentioned in the vid, need to explore more), we maximize 
L2  = log(D(G(z))) is equivalent to minimizing L1


Summary of basic GAN training

generate fake images using gen
pass the fake images through disc - get predictions, which in disc perspecitve should be fake for it to be a good disc, zero labels.
take the real images, pass through disc - get predictions, which in disc perspective should be real for it to be a good disc, one labels
loss1 = loss_fn(fakes_preds, zero_labels)
loss2 = loss_fn(real_preds, one_labels)

loss_d = 1/2(loss1 + loss2)
loss_d.backward(retain_graph = true) - as we will need the fakes generated to train generator.
update weights -- only discriminator weights are updated.

use the fakes data, pass through again as discriminator weights are updated.
get_preds fake_preds, from gen perspective we need a dumb discriminator, so it should predict ones.

loss_g = loss_fn(fake_preds, ones)
loss_g.backward()
opt_g.step() - only generator weights are updated


loss.backward(retain_graph = True) - retains the values of the nodes in computation graph. 
If False, which it is by default, it frees the memory.


