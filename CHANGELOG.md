# Changes Made

##### File Naming and Navigations (@iris0614 by Tuesday)
- Improvement in File naming
      - **Addressed**: https://github.com/UBC-MDS/fraud_detection/tree/0b73299fc8f71be769f3772f1903d696c2ced5b7
- Delete unnecessary files (PDF reports and extra.)
      - **Addressed**: https://github.com/UBC-MDS/fraud_detection/tree/ff0ed6f3b4928267134867ff5faa7edfe53f0069
- Create a dedicated `results` folder to store all CSV files and figures created from running the full scripts.
      - **Addressed**: https://github.com/UBC-MDS/fraud_detection/tree/8d2e0be9ff2274c20f7c50cd0f636b1f5acb9e8a

##### Script Improvements (@korayt by Thursday)
- Add additional metrics such as `AUC-ROC` curve and `Precision-Recall` curve to more clearly state the results and discuss more on the problems of our results.
      - **Addressed**: https://github.com/UBC-MDS/fraud_detection/tree/e85407102e5c36cc33db59bb20de4c975db7e9d9
            - Added `results/plots/precision_recall.png`.
- Change `pyteset` implementation. No tests seem to have ran when tried running `pytest` from root terminal.

##### Edit Figures (@shawnhu444 by Thursday)
- Change the name of the figure 1 and 2 as they seem to have the same name. 
- Add in title for figure 4.
      - All figure titles are now added along with figure numbers in Jupyter book. 
- Edit the subplots for figure 1 for easier interpretations. Add more columns rather than having it vertically linear in 1 column.
      - **Addressed**: https://github.com/UBC-MDS/fraud_detection/tree/64804a761a5b6e0b8cab9b06529e1ce3abbd4845

##### README Improvements (@jlee2843 by Thursday)
- [ ] No File Consumption Order section. 
- [ ] Distinguish between different notebooks that exists between `notebooks` folder. 

#### Discarded Improvements:
- Reduce repeated functions. For example: `download_data.py` should import `load_data` from `load_data.py` instead of redefining the function `load_data` inside `download_data.py`.
      - Reason for discarding: `download_data.py` and `load_data` function within `src` folder are written based on two different data formats; reusing `load_data` function instead of creating another script, `download_data.py`, is feasible.
