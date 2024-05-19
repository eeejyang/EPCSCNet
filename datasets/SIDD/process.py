from PIL import Image
import scipy.io as sio
import glob
from utils import *

# save the validation set as png images
gt = sio.loadmat('val_gt.mat')['ValidationGtBlocksSrgb']
noisy = sio.loadmat('val_noisy.mat')['ValidationNoisyBlocksSrgb']
num_image, num_patch = gt.shape[:2]
for i in range(num_image):
    save_dir = f"./val/{1000-i:04d}"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for j in range(num_patch):
        sub_gt = gt[i, j]
        sub_noisy = noisy[i, j]
        Image.fromarray(sub_gt).save(f"{save_dir}/GT_p{j:04d}.png")
        Image.fromarray(sub_noisy).save(f"{save_dir}/noisy_p{j:04d}.png")


# slice the training images to 256x256
GT_files = sorted(glob.glob("./SIDD_Medium_Srgb/**/*GT*", recursive=True))
noisy_files = [f.replace("GT", "NOISY") for f in GT_files]
ps = 512
stride = 256


def worker(idx):
    gt_file = GT_files[idx]
    noisy_file = noisy_files[idx]
    # print(gt, noisy, sep='\n')
    save_dir = f"./train/{idx:04d}"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    gt = np.array(Image.open(gt_file))
    noisy = np.array(Image.open(noisy_file))

    H, W = gt.shape[:2]
    ind_h = list(range(0, H - ps, stride))
    if ind_h[-1] != H - ps:
        ind_h.append(H - ps)
    ind_w = list(range(0, W - ps, stride))
    if ind_w[-1] != W - ps:
        ind_w.append(W - ps)

    pos_list = [(h, w) for h in ind_h for w in ind_w]
    for patch_idx, (h, w) in enumerate(pos_list):
        sub_gt = gt[h : h + ps, w : w + ps, :]
        sub_noisy = noisy[h : h + ps, w : w + ps, :]
        Image.fromarray(sub_gt).save(f"{save_dir}/GT_p{patch_idx:04d}.png")
        Image.fromarray(sub_noisy).save(f"{save_dir}/noisy_p{patch_idx:04d}.png")


from multiprocessing import Pool

pool = Pool(12)
pool.map(worker, list(range(len(GT_files))))
pool.close()
pool.join()
