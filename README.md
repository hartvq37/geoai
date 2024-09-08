# OpenGeoAI : tools for GeoAI and remote sensing analysis (DEV)

Welcome to **OpenGeoAI (DEV)**! 
OpenGeoAI is a suite of tools and resources that provide access to published research and deep learning model implementations for geospatial tasks

## Paper and code
### Title: Improving individual tree detection and crown delineation of urban trees from aerial imagery with semi-supervised deep learning (under review)

Note: The current repo only contain necessary components for reproducing the results for reviewing purpose. 
<p align="left">
  <img src="https://github.com/user-attachments/assets/4223ed22-ad6b-407c-9f89-58350923cefa" width="200" height="auto"/>
</p>

| Training type  | Model | Tasks |
| ------------- | ------------- | ------------- |
| Supervised  | Mask-RCNN  | Individual Crwon Segementation  |
| Semi-supervised | Soft-Teacher (Mask-RCNN)  | Individual Crwon Segementation  |
| Supervised  | Faster-RCNN  | Individual Tree Detection  |
| Supervised  | RetinaNet  | Individual Tree Detection  |
| Semi-supervised  | MixPL (Faster-RCNN)  | Individual Tree Detection  |
| Semi-supervised  | Consistent-Teacher (RetinaNet)  | Individual Tree Detection  |

### Getting Started
The paper is currently under review. Upon acceptance, a complete package will be published and shared through this repository, along with a reference link to the published journal. All current code is integrated into Jupyter notebooks, enabling the implementation of all experimental models discussed in the paper. However, these notebooks are intended solely for paper review purposes at this stage.    

### Prerequisites
- Python 3.8+
- OpenMMlab
- geopandas
- rasterio
- plotly

### Installation


### Quick Start Guide


## Use Cases


## License

This project is licensed for non-commercial use at this stage.

The codes are under the CC BY-NC 4.0 License - see the [LICENSE](https://creativecommons.org/licenses/by-nc/4.0/) file for details.



