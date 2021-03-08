# ICP-MCC
ICP with Maximum Correntropy Criterion

Run the command to see the arguments.
```python
python icp_test.py -h
```

Run the command to run normal ICP
```python
python icp_test.py --input path/to/pointcloud1 --input2 path/to/pointcloud2 --metric mse --iter 30
```

Run the command to run ICP-BiMCC
```python
python icp_test.py --input path/to/pointcloud1 --input2 path/to/pointcloud2 --metric mcc --iter 30 --bi_dir
```
