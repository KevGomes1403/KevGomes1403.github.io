+++
date = '2025-09-20T15:16:45-04:00'
draft = true
title = 'Combined Perceptron Branch Predictor in gem5'
categories = ["technical"]
math = true
+++

Branch predictors are crucial in keeping a CPU's frontend fed with instructions by capturing the complexities inherent to a program's control flow. Neural networks hvae similarly captured complexities of the world at large within their parameters. Multi-layer perceptrons preceded CNNs and transformers, and while not being as computationally impressive, can learn non-linear functions and provide fast classification in constrained environments.

This paper from 2005 proposes a perceptron-based branch predictor combining global history and branch path information. I've sought out to implement it in gem5 and validate its results in a RISC-V model.

Given that this architecture was introduced two decades ago, even prior to TAGE, I'm curious how well it stacks up against other common architecture. This post is also an exploration of branch predictor performance on sample baremetal workloads as a complement to my coursework.

# Architecture

The predictor combines a history-based perceptron and an address-based one. It maintains the following data structures:

1. *Global Branch History Register* (`size: globalHistoryBits`), a register of outcomes for all previous branches.
2. *Branch History Table* (`size: branchAddrBits x localHistoryBits`), a table of outcomes for a particular branch, indexed by the lower-order bits of the PC.
3. *Path Table*. (`size: history_len x branchAddrBits`), containing the PC's of the `history_len` previous branches.
4. *History Weight Table* (`size: branchAddrBits x history_len+1`), the weights used to train the history-based perceptron.
5. *Address Weight Table* (`size: branchAddrBits x branchAddrBits+1`), the weights used to train the address-based perceptron.

The paper suggests using branch path information for selecting weights to improve accuracy. Hence, $i$-th entry of each Weight Table is indexed by the $i$-th entry of the Path Table, that is $w_i = WT[PT[i]]$. It uses the step function as its activation, and can be described by the following equations:

$$y = w_{hist} \cdot \begin{bmatrix} 1 & h \end{bmatrix}^T + w_{addr} \cdot \begin{bmatrix} 1 & x \end{bmatrix}^T$$
$$pred = step(y)$$

The pseudocode provided by the paper for prediction:
```C
/* Calculate the output of the
history-based perceptron */
y_hist=W_hist[PC][0];

for (j = 1; j <= HISTORY_LENGTH; j++) {
    k = PT[j-1];
    if (history_reg[j-1]) y_hist += W_hist[k][j];
    else y_hist -= W_hist[k][j];
}
/* Calculate the output of the
address-based perceptron */
y_addr=W_addr[PC][0];

for (j = 1; j <= N_ADDR_BITS; j++) {
    k = PT[j-1];
    if (PC[j-1]) y_addr += W_addr[k][j];
    else y_addr -= W_addr[k][j];
}

y = y_hist + y_addr;
if ( y >= 0 ) prediction = true;
else prediction = false;
```

For updating the predictor:
```C
if (last_prediction!=outcome || (last_y <= THETA && last_y >= -THETA)) {
    /* update the history-based perceptron */
    if (outcome) weight_inc(W_hist[PC][0]);
    else weight_dec(W_hist[PC][0]);

    for (j = 1; j <= HISTORY_LENGTH ; j++) {
        k = PT[j-1];
        if (outcome == hist[j-1]) weight_inc(W_hist[k][j]);
        else weight_dec(W_hist[k][j]);
    }

    /* update the address-based perceptron */
    if (outcome) weight_inc(W_addr[PC][0]);
    else weight_dec(W_addr[PC][0]);

    for (j = 1; j <= N_ADDR_BITS; j++) {
        k = PT[j-1];
        if (outcome == PC[j-1])
        weight_inc(W_addr[k][j]);
        else
        weight_dec(W_addr[k][j]);
    }
}
```
# Results

I evaluated the combined perceptron branch predictor against existing implementations for TAGE, a bimodal predictor, and a tournament predictor in gem5. The configuration for each predictor is listed in the following table.

| Branch Predictor     | Size     |
|----------------------|----------|
| Combined Perceptron  | Global History Bits: <br>Local History Bits: <br>Number of Address Bits:      |
| BiModal              | Global Predictor Entries: 8192<br>Choice Predictor Entries: 8192      |
| Tournament           | Local Predictor Entries: 2048<br>Global Predictor Entries: 8192<br>Choice Predictor Entries: 8192  |
| TAGE                 | Number of history tables: 7<br>Minimum history size of TAGE: 5<br>Maximum history size of TAGE: 130      |
