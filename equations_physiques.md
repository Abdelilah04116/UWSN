# 🔬 Équations Physiques - UWSN PPO

Ce document détaille les équations physiques utilisées dans le projet pour la modélisation de la propagation acoustique sous-marine et la consommation énergétique.

## 🌊 Propagation Acoustique Sous-marine

### 1. Vitesse du son (Équation de Mackenzie)

La vitesse du son dans l'eau de mer dépend de la température, de la salinité et de la profondeur :

```
c(T,S,D) = 1448.96 + 4.591*T - 5.304×10⁻²*T² + 2.374×10⁻⁴*T³
           + 1.340*(S-35) + 1.630×10⁻²*D + 1.675×10⁻⁷*D²
           - 1.025×10⁻²*T*(S-35) - 7.139×10⁻¹³*T*D³
```

**Variables :**
- `T` : Température (°C)
- `S` : Salinité (PSU - Practical Salinity Units)
- `D` : Profondeur (m, valeur positive)
- `c` : Vitesse du son (m/s)

**Plages typiques :**
- Température : 0-30°C
- Salinité : 30-40 PSU
- Profondeur : 0-1000 m
- Vitesse du son : 1450-1550 m/s

### 2. Absorption acoustique (Francois & Garrison)

Le coefficient d'absorption acoustique modélise l'atténuation du signal :

```
α(f,T,S,D) = A₁*P₁*f₁*f²/(f₁² + f²) + A₂*P₂*f₂*f²/(f₂² + f²) + A₃*P₃*f²
```

**Coefficients :**
- `A₁ = 0.106 * exp((T - 26) / 9)`
- `A₂ = 0.52 * (1 + T / 43) * (S / 35)`
- `A₃ = 0.00049 * exp(-(T / 27 + D / 17))`

**Fréquences de relaxation :**
- `f₁ = 0.78 * √(S/35) * exp(T / 26)`
- `f₂ = 42 * exp(T / 17)`

**Facteurs de pression :**
- `P₁ = 1`
- `P₂ = 1 - 1.37×10⁻⁴*D + 6.2×10⁻⁹*D²`
- `P₃ = 1 - 3.84×10⁻⁴*D + 7.57×10⁻⁸*D²`

**Variables :**
- `f` : Fréquence (kHz)
- `α` : Coefficient d'absorption (dB/km)

### 3. Perte de trajet (Path Loss)

La perte totale de trajet combine la perte géométrique et l'absorption :

```
TL = 20*log₁₀(d) + α*d/1000
```

**Variables :**
- `TL` : Perte de trajet (dB)
- `d` : Distance (m)
- `α` : Coefficient d'absorption (dB/km)

**Composantes :**
- **Perte géométrique** : `20*log₁₀(d)` - Atténuation due à la propagation sphérique
- **Perte d'absorption** : `α*d/1000` - Atténuation due aux propriétés de l'eau

## ⚡ Modèle Énergétique

### 1. Énergie de transmission

L'énergie nécessaire pour transmettre des données dépend de la distance :

```
E_tx = (E_elec + E_amp * d²) * k
```

**Variables :**
- `E_tx` : Énergie de transmission (J)
- `E_elec` : Énergie électronique (J/bit)
- `E_amp` : Énergie d'amplification (J/bit/m²)
- `d` : Distance (m)
- `k` : Taille des données (bits)

**Valeurs typiques :**
- `E_elec = 50 nJ/bit` - Consommation de l'électronique
- `E_amp = 1 pJ/bit/m²` - Consommation de l'amplificateur

### 2. Énergie de réception

L'énergie pour recevoir des données :

```
E_rx = E_elec * k
```

**Variables :**
- `E_rx` : Énergie de réception (J)
- `E_elec` : Énergie électronique (J/bit)
- `k` : Taille des données (bits)

### 3. Énergie en veille

L'énergie consommée en mode veille :

```
E_idle = E_idle * t
```

**Variables :**
- `E_idle` : Énergie en veille (J)
- `E_idle` : Consommation en veille (J/s)
- `t` : Temps (s)

**Valeurs typiques :**
- `E_idle = 1 μJ/s` - Consommation en veille

## 🎯 Fonction de Récompense PPO

La fonction de récompense guide l'apprentissage de l'agent PPO :

```
reward = -E_total/1000 + (-d/1000) + 100*success + penalties
```

**Composantes :**

1. **Minimisation énergétique** : `-E_total/1000`
   - Encourage la minimisation de la consommation énergétique
   - Normalisation par 1000 pour équilibrer les ordres de grandeur

2. **Pénalité de distance** : `-d/1000`
   - Décourage les longs trajets
   - Normalisation par 1000

3. **Récompense de succès** : `100*success`
   - Forte récompense (+100) si la destination est atteinte
   - `success = 1` si destination atteinte, `0` sinon

4. **Pénalités additionnelles** :
   - **Boucles** : `-5` pour revisiter un nœud
   - **Énergie faible** : `-10` pour nœuds avec énergie < 200J
   - **Étapes longues** : `-1` si trop d'étapes
   - **Proximité** : `+proximité_amélioration/100`

## 📊 Métriques de Performance

### 1. Taux de succès
```
Success Rate = (Épisodes réussis) / (Total épisodes) × 100%
```

### 2. Consommation énergétique moyenne
```
Energy = (1/N) × Σ(E_i)
```
où `E_i` est l'énergie consommée dans l'épisode `i`

### 3. Latence moyenne
```
Latency = (1/N) × Σ(L_i)
```
où `L_i` est la latence de l'épisode `i`

### 4. Efficacité énergétique
```
Efficiency = Energy / (Data Size × Distance)
```
Énergie par bit transmis par mètre

## 🔬 Validation des Modèles

### 1. Vitesse du son
- **Validation** : Comparaison avec données océanographiques
- **Plage** : 1450-1550 m/s pour eau de mer typique
- **Précision** : ±1 m/s dans les conditions normales

### 2. Absorption acoustique
- **Validation** : Mesures expérimentales en mer
- **Plage** : 0.1-10 dB/km selon fréquence et conditions
- **Précision** : ±20% dans la plupart des cas

### 3. Consommation énergétique
- **Validation** : Comparaison avec capteurs réels
- **Plage** : 50-500 μJ pour transmission typique
- **Précision** : ±10% pour distances < 1km

## 📚 Références

1. **Mackenzie, K. V.** (1981). "Nine-term equation for sound speed in the oceans." *The Journal of the Acoustical Society of America* 70.3: 807-812.

2. **Francois, R. E., and G. R. Garrison** (1982). "Sound absorption based on ocean measurements. Part II: Boric acid contribution and equation for total absorption." *The Journal of the Acoustical Society of America* 72.6: 1879-1890.

3. **Akyildiz, I. F., et al.** (2005). "Underwater acoustic sensor networks: research challenges." *Ad hoc networks* 3.3: 257-279.

4. **Heidemann, J., et al.** (2012). "Underwater sensor networks: applications, advances and challenges." *Philosophical Transactions of the Royal Society A* 370.1958: 158-175.

---

*Ces équations sont implémentées dans le code Python du projet et peuvent être ajustées selon les conditions spécifiques de l'environnement sous-marin.*
