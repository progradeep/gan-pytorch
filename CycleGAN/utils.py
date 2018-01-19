import os
import matplotlib.pyplot as plt

def save_imgs(realA, fakeB, cycleA, realB, fakeA, cycleB, save_dir, epoch,step):
    data = [realA, fakeB, cycleA, realB, fakeA, cycleB]
    names = ["realA", "fakeB", "cycleA", "realB", "fakeA", "cycleB"]

    fig = plt.figure()

    for i, d in enumerate(data):
        d = d.squeeze()
        im = d.data.cpu().numpy()
        im = (im.transpose(1,2,0)+1)/2

        f = fig.add_subplot(2,3,i+1)
        f.imshow(im)
        f.set_title(names[i])
        f.set_xticks([])
        f.set_yticks([])


    p = os.path.join(save_dir, "epoch-%s_step-%s.png" % (epoch,step))
    plt.savefig(p)