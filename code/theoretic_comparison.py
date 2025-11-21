import numpy as np
import matplotlib.pyplot as plt
import os

# alpha in [0, 0.5)
alphas = np.linspace(0.0, 0.4999, 500)

def R_honest(alpha):
    return alpha

def R_original_selfish(alpha, gamma):
    a = alpha
    num = a*(1-a)**2*(4*a + gamma*(1-2*a)) - a**3
    den = 1 - a*(1 + (2-a)*a)
    return num / den

def R_modified(alpha, gamma):
    a = alpha
    num = a * (a**3*gamma - 3*a**2*gamma + a**2 + 3*a*gamma - 2*a - gamma)
    den = a**4 - 2*a**3 + a - 1
    return num / den

gammas = [1.0, 0.5, 0.0]  # 큰 값부터 작은 값 순서로
# 학술 논문용 색맹 친화적 색상 팔레트 (파란색 계열 그라데이션)
colors = ['#2166ac', '#4393c3', '#92c5de']  # 진한 파랑 → 중간 파랑 → 밝은 파랑

plt.figure(figsize=(10, 10))

# Honest mining line (gamma-independent) - 검은색 실선
plt.plot(alphas, R_honest(alphas),
         color='k', linestyle='-', linewidth=2.5, 
         label="Honest mining")

for color, gamma in zip(colors, gammas):
    R_orig = R_original_selfish(alphas, gamma)
    R_mod  = R_modified(alphas, gamma)

    # Selfish mining: 실선, 더 두껍게
    plt.plot(alphas, R_orig,
             color=color, linewidth=2.5,
             label=f"Selfish mining γ={gamma}")
    # Modified strategy: 점선, 같은 색상, 약간 얇게
    plt.plot(alphas, R_mod,
             color=color, linewidth=2,
             alpha=0.85, linestyle=(0, (8, 4)),
             label=f"Modified strategy γ={gamma}")
    
    # gamma=0일 때 selfish mining과 modified strategy 사이 영역 음영처리 (공격자의 추정 수익구간)
    if gamma == 0.0:
        plt.fill_between(alphas, R_mod, R_orig, 
                        color=color, alpha=0.15, 
                        label="Estimated profit region (γ=0)")

plt.xlabel("Miner hash power", fontsize=16)
plt.ylabel("Revenue ratio", fontsize=16)
#plt.title("Selfish mining and Modified strategy")
plt.xlim(0, 0.5)
plt.ylim(0, 1.0)

# Qubic's average hashrate share (22.09% = 0.2209)
# 수직선: 진한 회색 점선
qubic_alpha = 0.2209
plt.axvline(x=qubic_alpha, color='#666666', linestyle=':', linewidth=2, 
             label="Qubic's average hashrate share")

# 수직선 지점에 값 표시 (plot 내부, 오른쪽 위에 배치)
plt.text(qubic_alpha + 0.01, 0.04, '22.09%', 
         ha='left', va='top', fontsize=12, color='#666666',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#666666', alpha=0.8))

# Qubic's average hashrate share during selfish mining (28.02% = 0.2802)
# 수직선: 진한 회색 점선
qubic_selfish_alpha = 0.2802
plt.axvline(x=qubic_selfish_alpha, color='black', linestyle=':', linewidth=2, 
             label="Qubic's selfish mining hashrate share")

# 수직선 지점에 값 표시 (plot 내부, 오른쪽 위에 배치)
plt.text(qubic_selfish_alpha + 0.01, 0.04, '28.02%', 
         ha='left', va='top', fontsize=12, color='#666666',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#666666', alpha=0.8))

plt.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
plt.legend(loc='upper left', fontsize=12, framealpha=0.95)

fname = "fig/theoretical_comparison.pdf"
plt.savefig(fname, bbox_inches="tight", dpi=300)
plt.show()

print("Saved combined plot to:", fname)
