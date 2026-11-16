"""usage: equiv_ckpt.py <release|research> <repo_root> <ckpt_root> <npz> -> one line per run: hashes of output and grads"""
import sys, os, json, hashlib
mode, root, ckroot, npz = sys.argv[1:5]
if mode == "release":
    sys.path.insert(0, root); from pointwno import _deps; _deps.activate()
bm = os.path.join(root, "code/revision_2026/pointwno_v2_shock_benchmark_r1")
sys.path[:0] = [os.path.join(bm, "naca"), bm]
import train_naca_bm as TN, torch, numpy as np
import wavelet_convolution_v3 as WC
print("wavelet_convolution_v3", hashlib.sha256(open(WC.__file__, "rb").read()).hexdigest()[:12], flush=True)
torch.set_num_threads(int(os.environ.get("THREADS", "8")))
r = TN.OS.load_allpoints_npz(npz, knn_k=6)
inputs, targets, masks, knn = r[0], r[1], r[2], r[3]
g = 119                                                      # validation airfoil (shock case)
xyz, y, m, k = (torch.from_numpy(a[g][None]) for a in (inputs, targets, masks, knn))
xyz, y, m, k = xyz.float(), y.float(), m.float(), k.long()
grid = TN.make_internal_vrg(128, 96, stats=None, device=torch.device("cpu"))
for arm in os.environ.get("ARMS", "V2J2,FNOM").split(","):
    for s in [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]:
        rid = f"BM_NACA_{arm}_s{s}"
        cfg = json.load(open(os.path.join(ckroot, rid, "run_meta.json")))["provenance"]["config"]
        model = TN.build_model(cfg)
        sd = torch.load(os.path.join(ckroot, rid, "pcwno_best.pt"), map_location="cpu", weights_only=False)
        TN.prime_buffers(model, sd); model.load_state_dict(sd, strict=True); model.eval()
        tr = TN.OS.MaskedGradTrainer(model, lr=1e-3, grad_weight=0.0)
        tr.opt.zero_grad()
        pred = model(xyz, grid)
        loss = TN.OS.masked_mse_loss(pred, y, m); loss.backward()
        hg = hashlib.sha256()
        for n, p in sorted(model.named_parameters()):
            hg.update(n.encode()); hg.update(p.grad.detach().reshape(-1).clone().numpy().tobytes())
        ho = hashlib.sha256(pred.detach().reshape(-1).clone().numpy().tobytes()).hexdigest()
        np.savez(os.path.join(os.environ["GRAD_OUT"], rid + ".npz"), **{n: p.grad.detach().numpy() for n, p in model.named_parameters()})
        print(f"{rid} params {TN.count_real(model)} loss {loss.item():.10e} out {ho[:16]} grad {hg.hexdigest()[:16]}", flush=True)
