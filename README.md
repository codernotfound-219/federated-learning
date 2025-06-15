### **Client Selection Strategy for Federated Learning (CIFAR-10, 12 Clients)**  
**Objective**: Maximize accuracy gain per unit time by selecting 3 clients per round based on:  
1. **Training efficiency** (time/memory per batch).  
2. **Label diversity** (class imbalance mitigation).  
3. **Sample size** (clients with more data contribute more).  
4. **Historical accuracy contribution** (track past performance).  

---

### **Step 1: Quantify Key Factors**  
#### **1. Training Speed (Efficiency)**  
- Use `batch_size_ablation.csv` to compute **avg. time per batch** for each client:  
  \[
  \text{TimePerBatch}_i = \frac{\sum \text{training\_time}}{\sum \text{epoch}}
  \]  
- **Normalize** to [0, 1] (lower = faster).  

#### **2. Label Diversity (Class Balance)**  
- Use `client_label_distribution_int.csv` to compute:  
  - **Entropy-based score**: Higher entropy = more balanced labels.  
    \[
    H_i = -\sum_{l=0}^9 p_l \log p_l \quad \text{(where } p_l = \text{label}_l / \text{num\_items)}
    \]  
  - **Label scarcity**: Prioritize clients with rare labels (e.g., `label_2` in `part_2`).  

#### **3. Sample Size**  
- Weight clients by their data volume:  
  \[
  \text{SizeWeight}_i = \frac{\text{num\_items}_i}{\max(\text{num\_items})}
  \]  

#### **4. Historical Contribution**  
- Track each client’s **accuracy delta** (Δacc) in past rounds.  
- Use exponential moving average (EMA) to update:  
  \[
  \text{HistoricalScore}_i = \alpha \cdot \Delta\text{acc}_i + (1-\alpha) \cdot \text{HistoricalScore}_i
  \]  

---

### **Step 2: Composite Scoring Function**  
Combine factors into a **selection score** for client \(i\):  
\[
\text{Score}_i = \underbrace{\beta_1 \cdot (1 - \text{TimePerBatch}_i)}_{\text{Speed}} + \underbrace{\beta_2 \cdot H_i}_{\text{Diversity}} + \underbrace{\beta_3 \cdot \text{SizeWeight}_i}_{\text{Sample Size}} + \underbrace{\beta_4 \cdot \text{HistoricalScore}_i}_{\text{Contribution}}
\]  
**Weights**: Tune \( \beta_1, \beta_2, \beta_3, \beta_4 \) (e.g., 0.3, 0.2, 0.2, 0.3).  

---

### **Step 3: Adaptive Selection Algorithm**  
1. **Per-Round Selection**:  
   - Rank clients by `Score_i`.  
   - **Top 3**: Select fastest + most diverse + historically strong.  
   - **Stochasticity**: Introduce 10% randomness to explore suboptimal clients.  

2. **Label-Aware Adjustment**:  
   - If a label (e.g., `label_2`) is underrepresented globally:  
     - Boost scores of clients holding that label (e.g., `part_2`).  

3. **Dynamic Weighting**:  
   - Adjust \( \beta \) weights based on round progress:  
     - Early rounds: Favor **diversity** (\( \beta_2 \uparrow \)).  
     - Late rounds: Favor **speed/accuracy** (\( \beta_1, \beta_4 \uparrow \)).  

---

### **Example Round**  
| Client | TimeNorm | Entropy | SizeWeight | HistScore | **Total Score** |  
|--------|----------|---------|------------|-----------|-----------------|  
| part_0 | 0.8      | 1.2     | 0.9        | 0.7       | **3.0**         |  
| part_2 | 0.6      | 0.5     | 1.0        | 0.5       | **2.3**         |  
| part_7 | 0.7      | 1.5     | 0.8        | 0.9       | **3.2**         |  
**Selected**: `part_7`, `part_0`, `part_2` (top 3 + `part_2` for rare labels).  

---

### **Implementation Notes**  
1. **Data Structures**:  
   - Maintain a DataFrame with client scores, updated per round.  
2. **Efficiency**:  
   - Precompute label distributions and training speeds.  
3. **Fairness**:  
   - Cap selections per client to avoid overfitting to fast-but-narrow clients.  

This strategy balances **speed**, **diversity**, and **accuracy** while adapting to label imbalances. For code, use libraries like `numpy` for scoring and `pandas` for data handling.  

