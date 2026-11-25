# κ 敏感性复查（v3 证书，2026-09-02）

## (a) κ₂ 细梯度（3.5–7.0，步长 0.25）

| κ₂ | #VALID | 2018/2021/2024 | 新进入的 cell |
|---|---|---|---|
| 3.50 | 11 | 11/0/0 | CA×bn×2018, CA×semapop_vae×2018, CA×tabddpm×2018, CA×ttvae×2018, OH×bn×2018, OH×semapop_vae×2018, OH×ttvae×2018, TX×bn×2018, TX×semapop_vae×2018, TX×tabddpm×2018, TX×ttvae×2018 |
| 3.75 | 11 | 11/0/0 | – |
| 4.00 | 12 | 11/0/1 | OH×ttvae×2024 |
| 4.25 | 15 | 11/0/4 | OH×bn×2024, OH×semapop_vae×2024, TX×ttvae×2024 |
| 4.50 | 18 | 11/0/7 | CA×semapop_vae×2024, TX×bn×2024, TX×semapop_vae×2024 |
| 4.75 | 23 | 13/0/10 | CA×bn×2024, CA×tabddpm×2024, CA×ttvae×2024, OH×tabddpm×2018, OH×wgan_zcr×2018 |
| 5.00 | 23 | 13/0/10 | – |
| 5.25 | 26 | 14/0/12 | CA×wgan_zcr×2018, OH×tabddpm×2024, TX×tabddpm×2024 |
| 5.50 | 28 | 14/0/14 | CA×wgan_zcr×2024, OH×wgan_zcr×2024 |
| 5.75 | 28 | 14/0/14 | – |
| 6.00 | 31 | 14/3/14 | OH×bn×2021, OH×semapop_vae×2021, OH×ttvae×2021 |
| 6.25 | 31 | 14/3/14 | – |
| 6.50 | 31 | 14/3/14 | – |
| 6.75 | 32 | 14/3/15 | CA×flowsyn×2024 |
| 7.00 | 33 | 14/4/15 | TX×ttvae×2021 |

## (b) κ₁,inv（invalid-mass 容忍度）梯度，κ₂=5 固定

| κ₁,inv | L1 拒 | #VALID | 按模型 valid | max R_2^MA (valid) | 判定 |
|---|---|---|---|---|---|
| 0.005 | 27 | 23 | bn 6, semapop_vae 6, ttvae 6, tabddpm 4, wgan_zcr 1 | 1.043 | DEFER |
| 0.0075 | 27 | 23 | bn 6, semapop_vae 6, ttvae 6, tabddpm 4, wgan_zcr 1 | 1.043 | DEFER |
| 0.01 | 24 | 23 | bn 6, semapop_vae 6, ttvae 6, tabddpm 4, wgan_zcr 1 | 1.043 | DEFER |
| 0.015 | 18 | 27 | bn 6, semapop_vae 6, ttvae 6, ctgan 4, tabddpm 4, wgan_zcr 1 | 1.043 | DEFER |
| 0.02 | 15 | 29 | bn 6, ctgan 6, semapop_vae 6, ttvae 6, tabddpm 4, wgan_zcr 1 | 1.043 | DEFER |
| 0.03 | 15 | 29 | bn 6, ctgan 6, semapop_vae 6, ttvae 6, tabddpm 4, wgan_zcr 1 | 1.043 | DEFER |

## (c) B₀ 梯度（post-projection d_A 口径，κ 固定）

| B₀ | CA 2018 | OH 2018 | TX 2018 | OH 2024 | TX 2024 | 可部署场景数 |
|---|---|---|---|---|---|---|
| 0.15 | Defer | Deploy | Deploy | Defer | Defer | 2 |
| 0.20 | Defer | Deploy | Deploy | Defer | Defer | 2 |
| 0.25 | Deploy | Deploy | Deploy | Deploy | Deploy | 5 |
| 0.30 | Deploy | Deploy | Deploy | Deploy | Deploy | 5 |
| 0.35 | Deploy | Deploy | Deploy | Deploy | Deploy | 6 |
| 0.40 | Deploy | Deploy | Deploy | Deploy | Deploy | 6 |

D† per scenario: CA 2018 = 0.234, OH 2018 = 0.147, TX 2018 = 0.142, CA 2024 = 0.321, OH 2024 = 0.213, TX 2024 = 0.238

## (d) 刀口 cell

L2（|χ̄_H − cutoff|/cutoff < 15%，仅 L1 通过者）：

- TX×bn×2024: χ̄_H 4.279 vs cutoff 5.013 → pass
- TX×semapop_vae×2024: χ̄_H 4.334 vs cutoff 5.014 → pass
- CA×semapop_vae×2024: χ̄_H 4.479 vs cutoff 5.011 → pass
- OH×wgan_zcr×2018: χ̄_H 4.543 vs cutoff 5.043 → pass
- OH×tabddpm×2018: χ̄_H 4.545 vs cutoff 5.037 → pass
- CA×ttvae×2024: χ̄_H 4.574 vs cutoff 5.011 → pass
- CA×bn×2024: χ̄_H 4.651 vs cutoff 5.008 → pass
- CA×tabddpm×2024: χ̄_H 4.673 vs cutoff 5.011 → pass
- CA×wgan_zcr×2018: χ̄_H 5.066 vs cutoff 5.025 → fail
- TX×tabddpm×2024: χ̄_H 5.097 vs cutoff 5.016 → fail
- OH×tabddpm×2024: χ̄_H 5.110 vs cutoff 5.024 → fail
- OH×wgan_zcr×2024: χ̄_H 5.305 vs cutoff 5.027 → fail
- CA×wgan_zcr×2024: χ̄_H 5.345 vs cutoff 5.013 → fail

L1 invalid mass（0.5×–1.5× 容忍度 0.005）：

- CA×flowsyn: invalid 0.0028 (0.57×) → pass
- TX×flowsyn: invalid 0.0038 (0.76×) → pass
- OH×bn: invalid 0.0031 (0.63×) → pass
- OH×tabddpm: invalid 0.0048 (0.97×) → pass

L3（σ(H)）：τ_a=1.0 在 23 个 valid cell 中的 20 个处 binding（max_a χ_a^proj = 1.000），其余 3 个（OH×semapop_vae×2018, OH×ttvae×2018, OH×bn×2018）r* 本身已满足容差；τ<1 时仍可重新求解且可行（KL 升、nESS 降，见 l3_stress）。
