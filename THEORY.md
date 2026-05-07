# Theory — Methanol Synthesis on Cu/ZnO/Al₂O₃

A condensed, reader-friendly version of the full technical report. Use this if you want to understand *why* the model is built the way it is before diving into the code.

---

## 1. Industrial context

Methanol is one of the highest-volume bulk chemicals globally (>100 Mt/year), produced almost exclusively by catalytic hydrogenation of synthesis gas (CO + CO₂ + H₂) over copper-zinc oxide-alumina catalysts at 200–280 °C and 50–100 bar. The reactor is the most kinetically interesting unit operation in the plant — it must balance strong exothermicity against equilibrium constraints while removing heat at industrially viable rates.

The **Lurgi MRP** reactor (modelled here) is a vertical shell-and-tube heat exchanger with several thousand catalyst-filled tubes; saturated boiling water on the shell side maintains tubes nearly isothermal at 235–265 °C. Single-pass conversion is moderate (40–70 % of carbon) and unconverted gas is recycled.

---

## 2. Reaction network

Two parallel methanol-formation reactions plus the water-gas shift (WGS):

$$\text{R1:} \quad \text{CO} + 2\text{H}_2 \rightleftharpoons \text{CH}_3\text{OH}, \qquad \Delta H^\circ_{298} = -90.7\ \text{kJ/mol}$$

$$\text{R2:} \quad \text{CO}_2 + 3\text{H}_2 \rightleftharpoons \text{CH}_3\text{OH} + \text{H}_2\text{O}, \qquad \Delta H^\circ_{298} = -49.5\ \text{kJ/mol}$$

$$\text{R3:} \quad \text{CO} + \text{H}_2\text{O} \rightleftharpoons \text{CO}_2 + \text{H}_2\ (\text{WGS}), \qquad \Delta H^\circ_{298} = -41.2\ \text{kJ/mol}$$

By Hess's law R1 = R2 + R3, so only two of the three are linearly independent. Modern frameworks (VBF, Nestler) treat R2 and R3 as the independent rate steps and recover R1 from their sum, justified by isotopic-labelling experiments (Chinchen et al. 1987) showing that the carbon source for surface methoxide intermediates on Cu/ZnO is almost exclusively CO₂.

The model also tracks four **byproduct reactions** (DME, CH₄, EtOH, PrOH) — their selectivities at industrial conditions on a fresh catalyst are <1 % each, but they serve as monitoring channels for selectivity drift.

---

## 3. Equilibrium thermodynamics

Equilibrium constants from **Graaf & Winkelman (2016)** — the de-facto standard re-evaluation of the original 1986 Graaf regression:

$$\log_{10} K_{p,1} = \frac{3066}{T} - 10.592 \quad [\text{bar}^{-2}]$$

$$\log_{10} K_{p,3} = \frac{2073}{T} - 2.029 \quad [\text{--}]$$

$$K_{p,2} = K_{p,1} \cdot K_{p,3} \quad [\text{bar}^{-2}]$$

At T = 513 K (240 °C):

| Reaction                              | $K_{eq}$           | Comment                                        |
| :------------------------------------ | :----------------: | :--------------------------------------------- |
| R1: CO + 2H₂ ⇌ MeOH                   | 2.48×10⁻³ bar⁻²    | Moderately favourable; equilibrium-limited     |
| R2: CO₂ + 3H₂ ⇌ MeOH + H₂O            | 2.41×10⁻⁵ bar⁻²    | Less favourable than R1 (more entropy loss)    |
| R3: CO + H₂O → CO₂ + H₂ (forward WGS) | 1.03×10²            | Strongly favours products at this T            |

Reaction enthalpies are temperature-corrected via Kirchhoff's law using NIST Shomate-equation Cp polynomials.

---

## 4. Kinetics — three published frameworks

### 4.1 Graaf et al. (1988) — dual-site LHHW

Rate equations (3 reactions: r1 CO hydrogenation, r2 reverse WGS, r3 CO₂ hydrogenation):

$$r_1 = \frac{k_1 K_{\text{CO}} \big( P_{\text{CO}} P_{\text{H}_2}^{1.5} - P_{\text{MeOH}}/(K_{eq,3} P_{\text{H}_2}^{0.5}) \big)}{(1+K_{\text{CO}}P_{\text{CO}}+K_{\text{CO}_2}P_{\text{CO}_2})(P_{\text{H}_2}^{0.5} + (K_{\text{H}_2\text{O}}/\sqrt{K_{\text{H}_2}})P_{\text{H}_2\text{O}})}$$

(similar for r2 and r3, with the same denominator).

Published Arrhenius parameters (Graaf 1988 Table 5):

| Parameter            | Pre-exp                 | Activation E (J/mol)       |
| :------------------- | :---------------------: | :------------------------: |
| $k_1$ (CO hydro)     | 4.89×10⁷ mol/(kg·s·bar^0.5)  | 113,000                    |
| $k_2$ (RWGS)         | 9.64×10¹¹ mol/(kg·s·bar)     | 152,900                    |
| $k_3$ (CO₂ hydro)    | 1.09×10⁵ mol/(kg·s·bar^0.5)  |  87,500                    |
| $K_{\text{CO}}$      | 2.16×10⁻⁵ bar⁻¹             | -46,800 (van't Hoff)       |
| $K_{\text{CO}_2}$    | 7.05×10⁻⁷ bar⁻¹             | -61,700                    |
| $K_{\text{H}_2\text{O}}/\sqrt{K_{\text{H}_2}}$ | 6.37×10⁻⁹ bar⁻⁰·⁵ | -84,000 |

Validity range: 15–50 bar, 210–245 °C. The published parameters underpredict modern Cu/ZnO/Al₂O₃ catalysts (Slotboom 2020 explicitly notes this).

### 4.2 Vanden Bussche & Froment (1996) — single-site formate route

Single dual-functional active site. CO₂ hydrogenation through formate intermediate is the rate-determining step for methanol; CO acts only via WGS:

$$r_{\text{MeOH}} = \frac{k_{\text{MeOH}} P_{\text{CO}_2} P_{\text{H}_2} \left[1 - \frac{P_{\text{H}_2\text{O}} P_{\text{MeOH}}}{K_{p,1} P_{\text{H}_2}^3 P_{\text{CO}_2}}\right]}{D^3}$$

$$r_{\text{RWGS}} = \frac{k_{\text{RWGS}} P_{\text{CO}_2} \left[1 - K_{p,3} \frac{P_{\text{H}_2\text{O}} P_{\text{CO}}}{P_{\text{CO}_2} P_{\text{H}_2}}\right]}{D}$$

with the surface-coverage denominator

$$D = 1 + \frac{K_{\text{H}_2\text{O}}}{K_8 K_9 K_{\text{H}_2}} \frac{P_{\text{H}_2\text{O}}}{P_{\text{H}_2}} + \sqrt{K_{\text{H}_2} P_{\text{H}_2}} + K_{\text{H}_2\text{O}} P_{\text{H}_2\text{O}}$$

Parameters from VBF Table 2 — verified against the original 276-point regression on ICI 51-2 catalyst.

### 4.3 Nestler et al. (2020) — fugacity-based Henkel form

Re-fit of Henkel's mechanism against Park's 114 experimental points, with rates expressed in **fugacities** computed from the SRK EOS:

$$r_{\text{CO}_2} = \frac{k_1 K_2\, f_{\text{CO}_2}\, f_{\text{H}_2}^{3/2}\, \mathrm{EQ}_1}{\mathrm{DEN}^2}$$

$$r_{\text{RWGS}} = \frac{k_2 K_2\, f_{\text{CO}_2}\, f_{\text{H}_2}\, \mathrm{EQ}_2}{\mathrm{DEN}^2}$$

with

$$\mathrm{DEN} = 1 + K_1 f_{\text{CO}} + K_2 f_{\text{CO}_2} + K_3 \sqrt{f_{\text{H}_2}}\, f_{\text{H}_2\text{O}}/f_{\text{H}_2}$$

Best validity for high-CO₂ feeds (Power-to-Methanol applications). Parameters from Nestler 2020 Table 4 with a single multiplicative SCALE factor calibrated against the 114-point dataset to compensate for a known dimensional inconsistency in the published rate equation.

---

## 5. Effectiveness factors

Industrial Cu/ZnO/Al₂O₃ pellets are 5–6 mm — large enough that intra-pellet diffusion limits the observed rate. The model uses literature-calibrated effectiveness factors (Lommerts, Graaf & Beenackers 2000):

$$\eta_{\text{synthesis}} = \min\!\left(1,\, 0.50 \cdot \sqrt{0.006/d_p}\right)$$

$$\eta_{\text{WGS}} = \min\!\left(1,\, 0.65 \cdot \sqrt{0.006/d_p}\right)$$

The √(d_p,ref/d_p) scaling is the analytical asymptote for first-order kinetics in spherical pellets at high Thiele modulus.

---

## 6. Real-gas thermodynamics

At industrial pressures (50–100 bar) with polar species (H₂O, MeOH), partial-pressure-based rates introduce ~10–15 % errors. The model applies fugacity coefficients $\varphi_i$ from cubic EOS at every solver step.

### Peng–Robinson (used by VBF and Graaf)

$$P = \frac{RT}{V-b} - \frac{a(T)}{V(V+b) + b(V-b)}$$

with $\alpha_i(T) = [1 + \kappa_i(1 - \sqrt{T/T_{c,i}})]^2$ and $\kappa_i = 0.37464 + 1.54226\omega_i - 0.26992\omega_i^2$.

### Soave–Redlich–Kwong (used by Nestler)

$$P = \frac{RT}{V-b} - \frac{a(T)}{V(V+b)}$$

with $m_i = 0.480 + 1.574\omega_i - 0.176\omega_i^2$.

At 240 °C, 75 bar, both EOS give $\varphi_{\text{H}_2\text{O}} \approx 0.92$, $\varphi_{\text{MeOH}} \approx 0.94$, $\varphi$ for non-polars within 1 % of unity — modest but non-negligible for the polar species in the equilibrium quotient.

---

## 7. Pressure drop

Ergun (1952), in mass-flux form:

$$-\frac{dP}{dz} = \frac{150 \mu_g G (1-\varepsilon)^2}{\rho_g d_p^2 \varepsilon^3} + \frac{1.75 G^2 (1-\varepsilon)}{\rho_g d_p \varepsilon^3}$$

The single-most-common bug in PFR codes is passing catalyst bulk density (≈1100 kg/m³) instead of gas density (≈50 kg/m³ at 75 bar) to this equation — the resulting ΔP is ~74 bar over 7 m of bed and the model becomes thermodynamically forbidden. Always use the gas-phase density.

---

## 8. Effective heat transfer

For the **cooled mode**, the overall tube-to-shell heat transfer coefficient combines the inside-film, wall-conduction, and shell-side boiling resistances:

$$\frac{1}{U} = \frac{1}{h_{\text{in}}} + \frac{\delta_{\text{wall}}}{k_{\text{wall}}} + \frac{1}{h_{\text{shell}}}$$

Inside-film coefficient from the **Zehner–Schlünder** packed-bed correlation; wall conduction from the steel tube; shell-side boiling typically $h_{\text{shell}} \approx 8\,000$ W/(m²·K) on saturated water. The overall U comes out to ~280 W/(m²·K) for industrial Lurgi MRP geometry — and predicts a hot-spot ~60 K above coolant, in line with industrial measurements.

---

## 9. Reading list (in order of priority)

1. **Bisotti et al. (2022)** — modern critical comparison of kinetic models. Start here.
2. **Slotboom et al. (2020)** — re-fits all major models against the same dataset; gives the residuals you'd expect.
3. **Vanden Bussche & Froment (1996)** — the canonical paper, very readable.
4. **Graaf & Winkelman (2016)** — the equilibrium-constant re-evaluation everyone uses.
5. **Nestler et al. (2020)** — most modern of the three; best for high-CO₂ feeds.
6. **Olah, Goeppert & Prakash, *The Methanol Economy*** — broader context of why methanol matters.

For a fully-cited deep-dive, see the technical report at `docs/Methanol_Digital_Twin_Technical_Report.docx`.
