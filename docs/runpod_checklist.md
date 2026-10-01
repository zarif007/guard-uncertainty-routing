# RunPod checklist

Tick these in order. `docs/runbook.md` explains *why* each step exists and what
the verdicts mean; this file is just the sequence, with the SSH parts spelled
out.

**The mental model:** a pod is a rented Linux computer with a GPU. It starts
empty. Your laptop is only the keyboard and screen — you SSH in and the
commands run *there*. Nothing syncs automatically. Git is the bridge.

Every line below is tagged **[LAPTOP]** or **[POD]**. If you are ever unsure
where you are, look at the shell prompt: `zarif@...` is your Mac, `root@...`
is the pod. Or run `nvidia-smi` — it only works on the pod.

---

## A. Before you rent anything — [LAPTOP]

- [ ] **A1.** Accept the gated dataset terms in a browser, logged into
      HuggingFace: <https://huggingface.co/datasets/allenai/wildguardmix>
      (auto-approves, but it must be clicked).

- [ ] **A2.** Create a HuggingFace token with **read** access at
      <https://huggingface.co/settings/tokens>. Copy it somewhere you can
      paste from later.

- [ ] **A3.** Create a GitHub **personal access token** (classic, `repo`
      scope) at <https://github.com/settings/tokens>. You will paste this when
      pushing results from the pod. Do **not** copy your SSH private key to
      the pod.

- [ ] **A4.** Copy your SSH public key to the clipboard:
      ```bash
      pbcopy < ~/.ssh/id_ed25519.pub
      ```

- [ ] **A5.** Paste it into RunPod → **Settings → SSH Public Keys**
      (account-level, not per-pod). **Do this before starting a pod** — the key
      is injected at boot, so adding it later needs a pod restart.

- [ ] **A6.** Make sure the repo on GitHub is current, because the pod clones
      from there:
      ```bash
      git status && git push
      ```

---

## B. Create the pod — [LAPTOP, in the browser]

- [ ] **B1.** **Network volume first.** Storage → Network Volumes → New.
      **60 GB** (100 GB if you will run Phase 4). Note which region it is in —
      network volumes are region-locked and the pod must be in the same region.

- [ ] **B2.** Deploy a pod in **that region**, attaching the volume at
      `/workspace`.

      | | Pick |
      |---|---|
      | GPU | 24 GB VRAM minimum (RTX A5000, RTX 4090). 48 GB (L40S, A6000) is comfier |
      | Type | **On-demand, not spot** — see B4 |
      | Template | A PyTorch or CUDA template. Any recent one is fine |

- [ ] **B3.** Check the template exposes **TCP port 22**. Without it you only
      get RunPod's limited proxy SSH (no file transfer). The web terminal
      always works as a fallback.

- [ ] **B4.** Understand the one-pod rule before you start. The cost model
      prices each guard by **measured latency**, and latency only means
      something within one machine. Every prediction row carries an `env_hash`
      over `(backend, gpu_name, driver_version, llama_cpp_version, processor)`.
      Score everything on **one pod, in one go**. If you lose it, `analyze.py`
      reports `MIXED` and the frontier is meaningless — you either re-score or
      fall back to `--cost-basis size`, which is cruder.

      Phases 1–3 are about 5.5 GPU-hours, so this is a $3–6 experiment. Do not
      save a dollar on a spot instance and lose the run.

---

## C. Connect — [LAPTOP terminal]

- [ ] **C1.** Open the pod's **Connect** panel in RunPod. Copy the **Direct
      TCP** command. It looks like:
      ```bash
      ssh root@213.xxx.xxx.xxx -p 40123 -i ~/.ssh/id_ed25519
      ```
      (If only "basic SSH" is offered, the template did not expose port 22.
      Use the web terminal instead — everything below still works.)

- [ ] **C2.** Run it. Say `yes` to the host-key prompt. Your prompt should now
      read `root@...`.

- [ ] **C3. Start tmux immediately.** Phase 3 is four hours; a closed lid or a
      wifi hiccup kills a plain SSH session and takes the run with it.
      ```bash
      tmux new -s thesis
      ```
      Detach with `Ctrl-b` then `d`. Reconnect later with
      `ssh ...` then `tmux attach -t thesis`.

**Everything from here is [POD].**

---

## D. Bootstrap — ~20 min

- [ ] **D1.** Clone into the volume, not the home directory. The container disk
      is wiped when the pod stops; `/workspace` is not.
      ```bash
      cd /workspace && git clone https://github.com/zarif007/guard-uncertainty-routing.git && cd guard-uncertainty-routing
      ```
      Use HTTPS, not SSH — your key is not on the pod and should not be.

- [ ] **D2.** Virtual environment:
      ```bash
      python3 -m venv .venv && source .venv/bin/activate
      ```

- [ ] **D3.** One-shot bootstrap. This points `HF_HOME` **and**
      `MODEL_WEIGHTS_DIR` at the volume, installs requirements, installs the
      correct llama.cpp build, and prints the environment fingerprint.
      ```bash
      VOLUME=/workspace bash scripts/setup_runpod.sh
      ```

- [ ] **D4.** **Write down the `env_hash`** it prints at the end. Every
      prediction file must carry it.

- [ ] **D5.** Never run `pip install llama-cpp-python` yourself, now or later.
      It gives a CPU-only wheel that runs the whole experiment on the pod's
      host CPU **without erroring**. The scores would be fine and every latency
      number would be wrong — and latency is what prices the frontier, so the
      headline result would be silently invalid.

- [ ] **D6.** HuggingFace login, pasting the token from A2:
      ```bash
      huggingface-cli login
      ```

---

## E. Two free gates, before spending GPU time

- [ ] **E1.** Preflight — checks the engine build, GPU offload, HF auth, every
      model filename, disk headroom, template fingerprints.
      ```bash
      python scripts/preflight.py --models guard-panel
      ```
      Want **`0 fail`**. Check two lines specifically:
      - **`weights directory` must not warn.** If it does, `setup_runpod.sh`
        did not take, `MODEL_WEIGHTS_DIR` is unset, and 35 GB of weights will
        land on the ephemeral disk and vanish on restart. Re-run D3.
      - **`engine GPU support` must be `True`.** If not: `FORCE=1 BACKEND=cuda
        bash scripts/install_engine.sh`.

- [ ] **E2.** Verify the gates against data whose answer is known by
      construction, in both directions. No GPU needed.
      ```bash
      python scripts/verify_selective.py
      ```
      Must be **13/13**. A gate that cannot fail is decoration — this is what
      makes the rest of the study mean anything.

- [ ] **E3.** Datasets. `harmbench` and `xstest` are committed to git; these
      three are fetched (~21 MB).
      ```bash
      python scripts/download_datasets.py --datasets toxicchat wildguardtest openai_moderation
      ```

- [ ] **E4.** Smoke test — confirm the machine measures what you think before
      committing GPU hours. Writes to `results/smoke/`, never touches the real
      prediction set.
      ```bash
      MODELS="reference" N=40 bash scripts/smoke_test.sh
      ```

---

## F. Phase 1 — free signals · ~10 min · **GO / NO-GO**

- [ ] **F1.**
      ```bash
      python scripts/run_phase.py --phase 1
      ```
      One guard (Llama-Guard-3-8B), `xstest` + `harmbench`, signals 1 and 2.

- [ ] **F2.** Read the verdicts:
      ```bash
      python -c "
      import json; d=json.load(open('results/tables/gates.json'))
      print('score range:', d['score_range']['status'], '-', d['score_range']['reason'])
      for k,v in d.items():
          if k.startswith('gate_'): print(f\"{v['gate']:>3}  {v['status']:<24} {v.get('reason','')}\")
      "
      ```

- [ ] **F3.** Decide, in this order:

      | Read | Continue if | Stop if |
      |---|---|---|
      | `score_range` | `USABLE` | `POLARIZED` — scores pinned at 0/1, nothing to rank |
      | **S1** | `NATIVE_USABLE` | `NATIVE_DEGENERATE` — **the pipeline is broken.** S1 replicates Safety-Flag, which found native confidence usable on every model tested. This is a bug, not a finding |
      | **S2** | any verdict | — |

      **Do not proceed past a failing S1.** Phase 3 is four hours.

      S2 is a result either way: `MARGIN_BETTER` means rank on the raw margin
      instead of the probability — a one-line change with measurable payoff.

- [ ] **F4.** Push the results (see section J).

---

## G. Phase 2 — perturbation · ~1 hr · **GO / NO-GO**

- [ ] **G1.**
      ```bash
      python scripts/perturb.py --model reference --dataset xstest --n-variants 6
      ```

- [ ] **G2.**
      ```bash
      python evaluation/analyze.py
      ```

- [ ] **G3.** Read **S4** — this is the decision that saves the most time:
      - **`INDEPENDENT`** → instability carries information the margin cannot
        express. Scale it in Phase 3.
      - **`REDUNDANT_WITH_MARGIN`** → a noisy restatement of a small margin at
        ~6x the compute. **Do not scale it.** Report it as a replication of
        Safety-Flag's Appendix D.1 negative result, which
        `preregistration.md` commitment 1d requires rather than burying.

- [ ] **G4.** Check the variant drop rate `perturb.py` prints. If most variants
      fail the semantics-preserved check, the signal is measuring paraphrase
      quality, not model uncertainty.

- [ ] **G5.** Push the results.

---

## H. Phase 3 — the full panel · ~4 hrs · **the result**

- [ ] **H1.** Make sure you are inside tmux. Then:
      ```bash
      python scripts/run_phase.py --phase 3
      ```
      Three guards x five datasets. Add `--resume` if the pod was interrupted,
      `--evict` if the volume is tight.

- [ ] **H2.**
      ```bash
      python evaluation/analyze.py --headline-budget 0.05
      ```

- [ ] **H3.** Read **A1** first — it is the crux — then **A2**. All four
      combinations are publishable; that is the design. The only bad verdict is
      `NOT_EVALUABLE` (too few errors, or no priced alternative), which means
      **scale the sample** — not reinterpret.

- [ ] **H4.** Main figure lands at `results/figures/oversight_frontier.png`.

- [ ] **H5.** Push the results. **Then** you may stop the pod (section K).

---

## I. Phase 4 — precision ladder · ~5 hrs · optional

- [ ] **I1.** Only feeds signal 5. Skip it if A1 and A2 resolved cleanly.
      Needs another 45.8 GB.
      ```bash
      python scripts/run_phase.py --phase 4 --evict
      ```

---

## J. Getting results back — [POD] then [LAPTOP]

`results/` is tracked by git (only `results/smoke/` is ignored) and the
prediction files are small.

- [ ] **J1. [POD]** After each phase:
      ```bash
      git add results && git commit -m "phase N: <what ran>" && git push
      ```
      Git will ask for a username and password. Username is `zarif007`;
      **password is the GitHub token from A3**, not your account password.

- [ ] **J2. [LAPTOP]**
      ```bash
      git pull
      ```
      Now the numbers are in your editor.

---

## K. Shutting down — [LAPTOP, in the browser]

- [ ] **K1.** Confirm every result you care about is pushed and pulled.

- [ ] **K2.** **Stop** the pod (not terminate). GPU billing stops; the network
      volume keeps your weights and datasets for ~$7/month, so restarting is
      fast.

- [ ] **K3.** Do analysis and figure iteration on your laptop from here —
      `analyze.py` needs no GPU and that is where the real time goes.

- [ ] **K4.** Be aware: if you restart the pod later and it lands on a host
      with a different driver, `env_hash` changes and new scoring cannot be
      mixed with old scoring. **Finish all scoring before stopping**, or accept
      a re-score.

---

## L. Write-up

- [ ] **L1.**
      ```bash
      cat results/tables/claims_to_evidence.csv
      ```
      **Never write a claim this table has not marked `LICENSED`.** It is
      generated, not curated. `S5` will read `NOT_TESTED` — that is correct,
      and `preregistration.md` Amendment 1 explains it.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Lost the SSH connection mid-run | Normal | Reconnect, `tmux attach -t thesis`. The run continued |
| Run died when the laptop slept | You were not in tmux | Restart with `--resume` |
| `Permission denied (publickey)` | Key added after pod boot | Restart the pod, or use the web terminal |
| Only "basic SSH" offered | Template does not expose TCP 22 | Use the web terminal, or redeploy on a template that does |
| Weights vanished after a restart | `MODEL_WEIGHTS_DIR` unset | Re-run D3 with `VOLUME=/workspace` |
| Latency far higher than expected | CPU-only llama.cpp build | `FORCE=1 BACKEND=cuda bash scripts/install_engine.sh` |
| `analyze.py` says `MIXED` | Scoring split across machines | Re-score on one pod, or `--cost-basis size` and say so in the write-up |
| Suspiciously good AURC | Ties, not ranking | Check `tie_fraction` in `signal_performance.csv`. Above 0.90 the signal is reporting input order |
| Volume filling up | Weights accumulating | `--evict` deletes each GGUF after it is scored |
| Pod preempted mid-run | Spot instance | `--resume` skips prompts already scored |

---

## Things that are never the answer

Repeated from `CLAUDE.md`, because this is where the temptation arrives:

1. **Do not relax a threshold to make a gate pass.** Report that it failed.
2. **Do not edit `preregistration.md` once real predictions exist.**
3. **Do not write an unlicensed claim**, however obvious it looks.
4. **Re-run `verify_selective.py` after touching `gates.py`, `selective.py`,
   `signals.py` or `allocation.py`.** It must stay at 13/13.
