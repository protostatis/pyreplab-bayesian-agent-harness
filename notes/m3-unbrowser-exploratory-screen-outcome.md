# M3 Unbrowser Exploratory Screen Outcome

Date: 2026-08-11

Status: complete, exploratory only, permanently excluded

## 1. Scope and boundary

This note records the post-no-go exploratory screens run during the temporary
`ubuntu-local` compute window. It is not an amendment to the frozen
preregistration or gate. The frozen headroom result remains **no-go**, Phase 2
remains locked, and none of these rows may be used for meta-training,
calibration, development, final evaluation, or a final model-selection claim.

All screens used only `meta_train` policy bundles, known templates, fresh or
previously declared `T_pilot` tasks, and the immutable registry at hash
`2d3b6c3d956fed9d255782bef264f6333129e803fc6853b1fcebb4486a8a2d3f`.
No held policy or held template was consumed.

The executable source identity used by all three screens was
`e9d5dad028d60dd99c2801a12f1ab4c6f675679775fd793772fc8df13cebfca7`.
Stage 2 recorded code revision
`0a91e931ca8b96ccf48ce0e5ba4791029604c080`, dirty-worktree status hash
`33afffee5227758880e45a9f2ad0010eb6382f476ebe9aac94e91301b8dde7d4`,
and the pinned runtime identity in its preflight. The dirty snapshot was
intentional and bound by the exploratory preflight; it does not relax the
clean-worktree requirement of the frozen pilot path.

## 2. Immutable artifacts

| Screen | Manifest SHA-256 | Analysis SHA-256 | Panels | Attempts | Successes |
|---|---|---|---:|---:|---:|
| Wave 1A | `47b5a24b88e18cabda35e23aa2717357b8170d7060416350096650ad99e185a9` | `645e7afce0a1f900a5593e32b2b1b139767a47c901940ee53bac1108ee291d3c` | 8 | 96 | 24 |
| Wave 1B | `5218341321b5c0b8871463196f552b85178ce076032e159a9642ba413778338d` | `092799b313f06f5f9f3b3580b23e4d42e608d5ada8273ff3055deff9e068a890` | 8 | 40 | 15 |
| Stage 2 | `7e2c77c1d9920be4371ebd5dfe9a500e685a1ad85ac02aa2401cb868dc875a1c` | `2e372ce1ff17f9f81b11d20b3a9c6f08f75f0e16f91e64ea55cb55bce6405f2a` | 24 | 96 | 42 |

Stage 2 raw-result SHA-256:
`a69677bb91ffa2c09aeb1ba440359f17f30df228ce7e546ecdce6a2e7279b7f1`.

Stage 2 preflight SHA-256:
`14f4de167097fb51c1068f4b28c73e272f837bbb9cc33a021afa5e67d4d84148`.

All 40 panels completed. There were no infrastructure errors or model-runtime
failures. The three screens contain 232 attempts and 81 verified successes.

## 3. Exclusion audit

The remote deterministic dataset exporter was rerun against each complete run
root with the immutable treatment registry.

| Screen | Attempts found | Rows | `pilot_excluded` | Skipped |
|---|---:|---:|---:|---:|
| Wave 1A | 96 | 96 | 96 | 0 |
| Wave 1B | 40 | 40 | 40 | 0 |
| Stage 2 | 96 | 96 | 96 | 0 |
| **Total** | **232** | **232** | **232** | **0** |

Every result record also has task role `T_pilot`, the expected manifest hash,
status `completed`, and a unique panel coordinate. This audit preserves the
original no-go and all development/final boundaries.

## 4. Stage 2 neutral-task result

The two designed `distractor_recovery` tasks failed for every candidate and
replica. Removing them changes each denominator from 24 to 20 but does not
change any candidate's success count or rank.

Aliases used below:

- `text-submit`: `ub-decompose-text_first-submit_directly-diagnose_retry_once-expanded@2-439044f8`
- `structure-submit`: `ub-decompose-structure_first-submit_directly-diagnose_retry_once-expanded@2-e7e84c83`
- `text-reobserve`: `ub-decompose-text_first-final_reobserve-diagnose_retry_once-expanded@2-79cee6ac`
- `C-anchor`: `ub-brief_plan-structure_first-final_reobserve-fail_fast-expanded@2-3d2c51ee`

| Candidate | Neutral success | Replica 0 | Replica 1 | Discordant tasks | Both success | Both fail | Mean output tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| `text-submit` | 12/20 (60%) | 7/10 | 5/10 | 2/10 | 5/10 | 3/10 | 2647.25 |
| `structure-submit` | 11/20 (55%) | 6/10 | 5/10 | 3/10 | 4/10 | 3/10 | 2659.65 |
| `text-reobserve` | 10/20 (50%) | 5/10 | 5/10 | 2/10 | 4/10 | 4/10 | 3443.45 |
| `C-anchor` | 9/20 (45%) | 5/10 | 4/10 | 1/10 | 4/10 | 5/10 | 2914.85 |

Neutral repeat discordance is 8/40 policy-task cells (20%). The point winner
therefore is not a stable single-winner result. `text-submit` leads
`structure-submit` by one success, and the two bundles tie at 5/10 in replica
1.

### 4.1 Paired neutral outcomes

| `text-submit` comparison | `text-submit` only | Other only | Both success | Both fail | Net wins |
|---|---:|---:|---:|---:|---:|
| vs `structure-submit` | 3 | 2 | 9 | 6 | +1 |
| vs `text-reobserve` | 4 | 2 | 8 | 6 | +2 |
| vs `C-anchor` | 5 | 2 | 7 | 6 | +3 |

The paired counts support narrowing but are too small to establish a reliable
winner. In particular, the observation comparison has only five discordant
attempts.

### 4.2 Template consistency

Each cell has four attempts: two task seeds times two replicas.

| Candidate | Single-page | Multi-page | Search/filter | Table | Form | Recovery |
|---|---:|---:|---:|---:|---:|---:|
| `text-submit` | 4/4 | 4/4 | 3/4 | 1/4 | 0/4 | 0/4 |
| `structure-submit` | 4/4 | 4/4 | 0/4 | 2/4 | 1/4 | 0/4 |
| `text-reobserve` | 4/4 | 3/4 | 0/4 | 3/4 | 0/4 | 0/4 |
| `C-anchor` | 4/4 | 2/4 | 0/4 | 1/4 | 2/4 | 0/4 |

Single-page extraction is a ceiling. Designed recovery and hard form entry are
floors. The apparent specializations in search, table, and form are based on
only two seeds per template and must not be treated as allocator evidence.

## 5. Cross-wave stability

For context only, pooling neutral attempts across the screens where each
candidate appeared gives:

| Candidate | Neutral successes | Neutral attempts | Rate | Mean output tokens |
|---|---:|---:|---:|---:|
| `structure-submit` | 17 | 27 | 63.0% | 2640.11 |
| `text-submit` | 15 | 27 | 55.6% | 2638.22 |
| `text-reobserve` | 17 | 34 | 50.0% | 3208.41 |
| `C-anchor` | 11 | 27 | 40.7% | 3018.56 |

This is not a pooled estimator. Wave 1B reused Wave 1A tasks, and
`structure-submit` entered Stage 2 because it led that adaptive screen. The
fresh, fixed-before-outcome Stage 2 ordering reverses the first two rows by one
success. The defensible conclusion is a two-candidate family, not a declared
winner.

Across all screens, the neutral tasks produced 81/199 successes. Designed
recovery produced 0/33 successes and consumed relatively high output cost.

## 6. Factor-level interpretation

The manipulation failures require intention-to-treat interpretation.

- **Tool cap:** Wave 1A's expanded bundles achieved 19/48 successes versus
  5/48 for lean bundles. Both exact matched cap pairs improved by 2/8. This is
  the strongest screening signal, but expanded-cap compliance was only 43/48
  in Wave 1A and total Stage 2 compliance was 89/96. Keep expanded in the
  candidate family, but do not call the difference a clean cap effect.
- **Planning:** Wave 1A decompose bundles achieved 11/32 successes versus 7/32
  direct and 6/32 brief-plan, with 100% marker adherence. Keep decompose in the
  candidate family; the unbalanced screen still prevents a causal claim.
- **Verification:** Stage 2 `text-submit` beat its exact `text-reobserve` match
  12/20 to 10/20 on neutral attempts, with a 796-token lower mean. In Wave 1B,
  submit beat reobserve 6/8 to 1/8 for the structure label and tied 3/8 to 3/8
  for the text label. Final-reobserve adherence was 1/48 in Wave 1A and zero in
  Waves 1B and 2. Drop `final_reobserve` from the narrowed family.
- **Recovery:** Retry-labelled bundles had better marginal screening rates,
  but all 33 designed recovery attempts failed. Retain `diagnose_retry_once`
  only as part of the observed top bundle family; no recovery-effect claim is
  supported, and the recovery fixture/manipulation needs redesign.
- **Observation:** Stage 2 structure-first adherence was 4/48, compared with
  33/48 for text-first. The top two submit bundles differ by only one neutral
  success and about 12 mean output tokens. The nominal observation factor is
  unresolved and not behaviorally clean.

## 7. Decision

The exploratory screen narrows the viable bundle family to:

`decompose + submit_directly + diagnose_retry_once + expanded`

Retain both `text-submit` and `structure-submit` as exploratory candidates. Do
not declare either a statistically or operationally resolved winner. If one
bundle is needed as an engineering-smoke default before the mechanics are
reworked, use `text-submit`: it has the fresh-task point lead, slightly lower
neutral output cost, and much better nominal observation adherence. That choice
is provisional and is not a final model-selection decision.

Do not train or evaluate an allocator from these screens and do not unlock
Phase 2. Before spending more model compute:

1. Enforce observation behavior mechanically or remove the observation factor.
2. Restore 100% hard-cap compliance and test extension reload behavior.
3. Redesign the recovery probe so retry behavior can produce measurable task
   recovery.
4. Replace ceiling/floor tasks with discriminating fresh `T_pilot` seeds.
5. If another exploratory replication is justified after those fixes, freeze
   it before outcomes and compare only the two narrowed submit bundles with
   enough replicas to measure repeat stability.

The frozen gate report remains unchanged at
[`m3-headroom-pilot-e7f257c4-gate.json`](m3-headroom-pilot-e7f257c4-gate.json).
