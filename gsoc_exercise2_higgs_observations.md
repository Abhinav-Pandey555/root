# Exercise 2 - TMVA Higgs Classification Tutorial

## Tutorial: TMVA_Higgs_Classification.C

## Dataset
- Total: 20,000 events (10k signal + 10k background)
- Training: 14,000 events, Testing: 6,000 events
- Input variables: 7 physics measurements (m_jj, m_jjj, m_lv, m_jlv, m_bb, m_wbb, m_wwbb)

## Methods and Performance (ROC Score)
- DNN_CPU:    0.767 (best)
- BDT:        0.758
- Likelihood: 0.700
- Fisher:     0.654 (worst)

## Key Observation
- m_bb (b-tagged jet invariant mass) ranked #1 across all methods
- Most discriminating variable for Higgs signal vs background

## Neural Network Details
- Architecture: 5 layers, 4x64 neurons (Tanh) + 1 output (Linear)
- Optimizer: ADAM, Batch size: 128, Max epochs: 30
- Training time: ~8 seconds on CPU
- No significant overtraining detected

## Display Issue
- ROOT web canvas attempted on port 8801
- xdg-open not found in WSL2 environment
- All terminal output and generated files working correctly
