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
- **Format:** CSV, .mat, or other formats used
- **Number of Samples:** X
- **labels:** y
- **Collection Method:** A 5G communication setup is implemented and operated at 28 GHz radio frequency using USRPs and 5GCHAMPION testbed. The camera is synchronized with 5G CHAMPION testbed and is used to capture the camera data. PHY layer radio data such as CSI is collected using vector network analyzer connected to the 5G communication hardware. 
- **License:** MIT

## How to Use
### Downloading the Dataset
Clone this repository and access the dataset:
```bash
git clone https://github.com/psusarla95/28GHz_statichumanpositioning.git
```

### Loading the Dataset
Example usage in Python:
```python
X,y = load_data_from_scratch()
```

## Citation
If you use this dataset, please cite it as follows:
```
@INPROCEEDINGS{10646323,
  author={Zampato, Silvia and Susarlal, Praneeth and Jokinen, Markku and Tervo, Nuutti and Leinonen, Marko E. and López, Miguel Bordallo and Juntti, Markku and Rossi, Michele and Silvén, Olli},
  booktitle={2024 IEEE 4th International Symposium on Joint Communications & Sensing (JC&S)}, 
  title={Static Human Position Classification from Indoor mmWave Radio RSSI Measurements}, 
  year={2024},
  volume={},
  number={},
  pages={1-6},
  doi={10.1109/JCS61227.2024.10646323}
}

```

## License
MIT License

## Contact
For any issues or questions, contact [praneeth.susarla@oulu.fi], [silvia.zampato@phd.unipd.it].

