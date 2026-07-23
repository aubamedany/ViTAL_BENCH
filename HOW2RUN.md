Create python environment: 
```
conda create -n traffic python==3.11
```
```
python -m pip install -r requirements.txt
```
```
conda install -c conda-forge openjdk=21 -y
```

submit job:

condition: e0_oracle: give the agent gold chunks
           e4_agent: full pipeline

```
qsun run_oracle.sh
```