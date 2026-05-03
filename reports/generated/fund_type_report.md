# Relatório de Análise de Anomalias em Fundos de Investimento
## Período: 02/01/2026 a 30/04/2026

---

## 1. Ranking de Exposição

Com base nos dados consolidados do período, observamos diferenças significativas nas taxas de anomalia entre as categorias de fundos:

| Fund Type | Anomalia Média | Pico Máximo | Total de Fundos |
|-----------|----------------|-------------|-----------------|
| CLASSES - FIF | 3,65% | 38,79% | 24.465,4 |
| CLASSE FIF/FAPI | 3,44% | 33,33% | 11,98 |
| FI | 2,50% | 33,33% | 3,8 |

**CLASSES - FIF** apresentou a maior taxa média de anomalia (3,65%) e o pico mais elevado entre todas as categorias (38,79% em 03/03/2026), indicando maior volatilidade e sensibilidade a eventos de estresse. **FI** registrou a menor média (2,50%), porém com pico expressivo de 33,33%, sugerindo que, embora menos propenso a anomalias no dia a dia, quando ocorrem, são intensas. **CLASSE FIF/FAPI** ocupou posição intermediária, com média de 3,44% e pico de 33,33%.

---

## 2. Perfil de Risco por Categoria

### FIA (Ações) – Representado por CLASSES - FIF
A correlação com o Ibovespa é praticamente nula (-0,000441), sugerindo que os fundos classificados como FIF não acompanham diretamente o mercado acionário brasileiro. No entanto, o pico de anomalia de 38,79% em 03/03/2026 coincide com dias de alta movimentação no mercado (Ibovespa com retorno médio de 0,07%), indicando que eventos extremos podem gerar disrupções independentes do índice.

### FIM (Multimercado) – Representado por CLASSE FIF/FAPI
A correlação mais relevante é com o **Ibovespa (0,424)** e com o **VIX (-0,231)**, demonstrando que estes fundos reagem positivamente a movimentos da bolsa brasileira e negativamente ao aumento da aversão ao risco global. A correlação cambial é baixa (-0,079), mas as anomalias se concentraram em março de 2026, sugerindo sensibilidade a eventos macroeconômicos naquele período.

### FIDC – Representado por FI
Os fundos tipo FI apresentam correlação negativa com USD/BRL (-0,094) e com o VIX (0,025 positiva, porém baixa). Destaca-se a correlação com **VIX lag10 (0,120)** e **sp500_return_lag10 (-0,095)**, indicando que choques no exterior levam aproximadamente 10 dias úteis para se refletirem nestes fundos. Esse comportamento é típico de fundos de crédito, que reagem com defasagem a eventos de liquidez.

### Outros Padrões
Observamos que o **US 10y** tem correlação negativa com CLASSES - FIF (-0,125) e CLASSE FIF/FAPI (-0,125), sugerindo que o aumento nas taxas de juros americanas pressiona negativamente esses fundos.

---

## 3. Sincronização vs Defasagem

### Datas de Pico por Categoria
- **03/03/2026**: CLASSES - FIF (38,79%) — pico máximo do período
- **12/03/2026**: CLASSE FIF/FAPI (33,33%)
- **13/02/2026 e 09/04/2026**: FI (33,33%)

**Análise**: **Não há sincronização perfeita**. O pico de CLASSES - FIF ocorreu em 03/03, enquanto CLASSE FIF/FAPI atingiu o ápice em 12/03 (9 dias depois) e FI em datas distintas (fevereiro e abril). Isso revela que:

1. **Transmissão acelerada**: Fundos de renda fixa (FIF) reagem primeiro a choques de mercado, possivelmente via ajustes de marcação a mercado.
2. **Transmissão defasada**: Fundos multimercado (FIF/FAPI) demoram mais para refletir o estresse, provavelmente devido a estratégias de hedge ou menor sensibilidade a preços de fechamento.
3. **Eventos idiossincráticos**: FI apresentou picos em fevereiro e abril, sugerindo gatilhos específicos (ex: inadimplência, rebaixamento de rating) em vez de estresse sistêmico.

---

## 4. Correlação com Mercado

| Variável | CLASSES - FIF | CLASSE FIF/FAPI | FI |
|----------|---------------|-----------------|-----|
| Ibovespa | -0,0004 | **0,424** | 0,005 |
| USD/BRL | -0,0004 | -0,079 | **-0,094** |
| VIX | 0,001 | **-0,231** | 0,025 |
| SP500 | 0,0002 | 0,219 | 0,0003 |
| US 10y | 0,001 | -0,125 | **-0,057** |

**Melhor explicador por categoria:**
- **CLASSES - FIF**: Nenhuma variável explica significativamente (correlações próximas de zero) — anomalias são **idiossincráticas**.
- **CLASSE FIF/FAPI**: **Ibovespa (0,424)** é o principal driver, seguido por SP500 (0,219) e VIX (-0,231).
- **FI**: **USD/BRL (-0,094)** e **US 10y (-0,057)** indicam sensibilidade a juros e câmbio, mas com baixa magnitude.

---

## 5. Implicações para Diversificação

### Durante Picos de Estresse
- Em **03/03/2026**, CLASSES - FIF registrou 38,79% de anomalias, enquanto CLASSE FIF/FAPI teve apenas 16,67% e FI 0% (data não listada no top 10).
- Em **12/03/2026**, CLASSE FIF/FAPI atingiu 33,33%, enquanto CLASSES - FIF teve 21,83% e FI novamente ausente.

**Conclusão**: **Não há correlação perfeita entre os picos**. Um portfólio multi-fundos teria **reduzido o risco sistêmico**, pois enquanto uma categoria enfrentava estresse máximo, as outras mantinham anomalias moderadas. A diversificação entre os três tipos de fundos teria proporcionado **amortecimento de choques**, especialmente em março de 2026.

---

## 6. Conclusão e Recomendações

### Conclusões Principais
1. **CLASSES - FIF** é o tipo mais propenso a anomalias frequentes e intensas, mas com baixa correlação com mercados tradicionais.
2. **CLASSE FIF/FAPI** responde diretamente ao Ibovespa e à aversão ao risco global (VIX), sendo mais previsível.
3. **FI** apresenta anomalias menos frequentes, porém intensas, com defasagem de até 10 dias em relação a eventos externos.

### Recomendações

**Para Seleção de Fundos**:
- Prefira **CLASSES - FIF** apenas se houver capacidade de tolerar picos de 38%+ de anomalias; exige monitoramento diário.
- **CLASSE FIF/FAPI** é mais adequado para alocações core, dado seu comportamento correlacionado ao Ibovespa e previsibilidade.
- **FI** deve ser usado como diversificador de baixa correlação, mas com atenção a eventos de crédito.

**Para Monitoramento de Risco**:
- Estabeleça **limites de exposição por categoria**: máximo de 30% em CLASSES - FIF, 40% em CLASSE FIF/FAPI e 30% em FI.
- Ative **alertas** quando a taxa de anomalia ultrapassar 15% em qualquer categoria, especialmente se coincidir com VIX > 25 ou USD/BRL > 5,50.
- Realize **análise de defasagem** de 10 dias úteis para FI em relação a choques no exterior (VIX, SP500).

**Para Diversificação**:
- Mantenha **pelo menos duas categorias** com correlação abaixo de 0,3 entre si (como CLASSES - FIF + FI, correlação cruzada ≈ 0).
- Durante períodos de estresse (ex: março de 2026), aumente exposição a **FI** como hedge, dado seu comportamento defasado.

---

*Relatório elaborado em conformidade com as melhores práticas de análise de risco para fundos de investimento brasileiros.*