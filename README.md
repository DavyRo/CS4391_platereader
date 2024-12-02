# License Plate Reader

By Chhay Lay Heng, Kae Lewis, Davy Romine

### Overview
This program reads license plates from an included sample video. It then filters out low confidence results and compares them to a ground truth file.

### How to Run
Download all files. Run the following command:
```
pip install -r requirements.txt
```
Then, run `reader.py`.
To compare the results to `ground_truth.csv`, run `accuracy_test.py`.
