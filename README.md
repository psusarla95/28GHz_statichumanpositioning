# 28GHz_statichumanpositioning
Static Human Positioning in a passive sensing manner using CSI information from 28GHz communication inffrastructure setup. 

## File Structure
Describe the files included in this

```
📂 28GHz_statichumanpositioning/
├── 📄 README.md           # Documentation
├── 📂 dataset/            # Raw and processed data files
│   ├──convPNAtimestamps.m #used to convert time stamps in PNA data (not necessary for github dataset version)
|   ├──📂 pna_data/
│   │  ├── *.mat              # radio data from vector network analyzer. sm_sub[x]_[y].mat, x-subject number, y-empty room (er), humantracking (ht)
│   ├──📂 video_data/
|   |  ├── *.csv           #captured frame and timestamped camera data
|   |  ├── *.mp4           #captured video data 
├── 📂 statichumanpositioning/            # Code for data processing and analysis
│   ├── __init__.py      
│   ├── main.py            # main file containing all the steps from pre-processing to training and testing analysis of 5g data
|   ├── test_buildingdataset.py
|   ├── utils.py
├── 📄 LICENSE             # License for dataset usage
└── 📄 CITATION.cff        # Citation file for proper referencing
```

## Dataset Details
- **Format:** CSV, JSON, or other formats used
- **Number of Samples:** X
- **Number of Features:** Y
- **Collection Method:** Describe briefly
- **License:** Specify (e.g., CC BY 4.0, MIT, etc.)

## How to Use
### Downloading the Dataset
Clone this repository and access the dataset:
```bash
git clone https://github.com/yourusername/dataset-repo.git
```

### Loading the Dataset
Example in Python:
```python
import pandas as pd
df = pd.read_csv("data/processed_data.csv")
print(df.head())
```

## Citation
If you use this dataset, please cite it as follows:
```
@dataset{your_dataset,
  author = {Your Name},
  title = {Dataset Name},
  year = {2025},
  url = {https://github.com/yourusername/dataset-repo}
}
```

## License
Specify the license for dataset usage.

## Contact
For any issues or questions, contact [your.email@domain.com](mailto:your.email@domain.com).

