# Offline RL under Distribution Shift: Representation Collapse and Reward Recovery

Research code for studying representation collapse in offline reinforcement learning, reward recovery via adversarial inverse RL, and the connection to preference-based alignment in large language models.

The project follows a four-part progression:

1. **Representation collapse in offline RL** — diagnosing and fixing feature rank degradation under IQL training
2. **Eigenvalue regularisation** — recovering representational diversity via log-determinant regularisation on the feature covariance
3. **Reward recovery via adversarial IRL** — extracting transferable reward functions from expert demonstrations using AIRL
4. **DPO extension** — connecting adversarial reward recovery to preference-based LLM alignment via Direct Preference Optimisation

This public version includes code, configs, and evaluation videos. Generated datasets, training logs, and intermediate artifacts are excluded — all results are reproducible from the provided scripts and configs.

## Demo

| Baseline (no regularisation) | Eigenvalue Regularisation |
|:---:|:---:|
| [walker2d_baseline.mp4](assets/walker2d_baseline.mp4) | [walker2d_eigreg.mp4](assets/walker2d_eigreg.mp4) |

## Repository Layout

```text
configs/
  section2/       IQL baseline and eigenvalue regularisation configs
  section3/       AIRL reward learning configs
  section4/       DPO fine-tuning configs (beta=0.1, beta=0.5)

scripts/
  section2/       IQL training, representation diagnostics, collapse analysis
  section3/       AIRL training, reward analysis
  section4/       DPO training, dashboard generation

src/
  iql/            IQL agent, shared encoder, DR3 eigenvalue regularisation
  airl/           AIRL discriminator, reward/value decomposition, SAC policy
  dpo/            DPO loss (Bradley-Terry), HH-RLHF data pipeline
```

## Running the Experiments

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

### Section 2: IQL with Representation Collapse Diagnostics

Baseline (no eigenvalue regularisation):
```bash
python3 scripts/section2/train_iql.py \
  --config configs/section2/iql_baseline.yaml \
  --output-dir outputs/section2_baseline
```

With eigenvalue regularisation:
```bash
python3 scripts/section2/train_iql.py \
  --config configs/section2/iql_dr3.yaml \
  --output-dir outputs/section2_eigreg \
  --use-dr3
```

### Section 3: AIRL Reward Recovery

```bash
python3 scripts/section3/train_airl.py \
  --config configs/section3/airl.yaml \
  --output-dir outputs/section3_airl
```

### Section 4: DPO Fine-Tuning

Requires `transformers`, `datasets`, and `accelerate`:
```bash
pip install transformers datasets accelerate
```

Fine-tune Qwen2.5-0.5B-Instruct on Anthropic HH-RLHF with different KL penalties:
```bash
# Weak KL penalty (more policy drift)
python3 scripts/section4/train_dpo.py \
  --beta 0.1 \
  --output-dir outputs/section4_dpo_beta01

# Stronger KL penalty (less drift)
python3 scripts/section4/train_dpo.py \
  --beta 0.5 \
  --output-dir outputs/section4_dpo_beta05
```

Compare the two runs:
```bash
python3 scripts/section4/analyze_dpo_metrics.py \
  --metrics outputs/section4_dpo_beta01/train_metrics.csv \
  --metrics2 outputs/section4_dpo_beta05/train_metrics.csv \
  --label1 "β = 0.1" --label2 "β = 0.5" \
  --output-dir outputs/section4_dpo_comparison \
  --title "DPO: KL Penalty Comparison"
```

## Main Entry Points

| Script | Purpose |
|--------|---------|
| `scripts/section2/train_iql.py` | IQL training with optional eigenvalue regularisation |
| `scripts/section2/analyze_representation.py` | Representation diagnostics (effective rank, PCA, eigenvalue spectrum) |
| `scripts/section3/train_airl.py` | AIRL discriminator training for reward recovery |
| `scripts/section4/train_dpo.py` | DPO fine-tuning with Bradley-Terry preference loss |
| `scripts/section4/analyze_dpo_metrics.py` | DPO training dashboard (single run or comparison) |

## Notes for Reviewers

- The project is structured as a research pipeline: representation diagnosis → fix → reward recovery → alignment connection.
- Each section builds on the previous one conceptually; they share architectural patterns (shared encoders, log-ratio objectives) but operate on different domains.
- The DPO extension (Section 4) demonstrates the structural connection between adversarial reward recovery in RL and preference-based alignment in LLMs — the same problem of recovering latent reward from observed behaviour.
- All outputs are reproducible from the provided code and configs.
