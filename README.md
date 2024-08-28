# OpenGeoAI : python tools for GeoAI and remote sensing analysis

<p align="center">
  <img src="https://github.com/user-attachments/assets/f72a49f2-32aa-4cf2-ad19-30ba7c74e756" width="200" height="auto"/>
</p>

Welcome to **OpenGeoAI**! Our mission is to bridge the gap between open-source AI packages and geospatial applications. OpenGeoAI aims to provide GIS users with an easy-to-use tool that allows them to quickly set up and perform geospatial analyses by leveraging the latest advancements in AI technologies.

## Supported tasks
Tree detections
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

## Features

- **Easy Integration**: OpenGeoAI provides a simple and intuitive interface for integrating AI models with geospatial data, reducing the learning curve for GIS professionals.
- **Rapid Setup**: Get started quickly with pre-configured settings and environment setups tailored for GIS applications.
- **Wide Range of AI Techniques**: Leverage various AI models, including machine learning, deep learning, and computer vision, to solve geospatial problems, all powered by OpenMMLab.
- **Scalability**: Designed to handle large geospatial datasets efficiently, making it suitable for both small-scale and large-scale applications.
- **Extensibility**: OpenGeoAI is modular and allows users to extend its functionality by integrating new AI models and geospatial analysis tools.
- **Community-Driven**: OpenGeoAI encourages collaboration and contributions from the community to continuously improve and expand its capabilities.

## Getting Started

### Prerequisites

- Python 3.8+
- GIS software (e.g., QGIS, ArcGIS)
- AI/ML libraries (e.g., TensorFlow, PyTorch)
- Geospatial libraries (e.g., GeoPandas, Shapely)

### Installation

To install OpenGeoAI, you can clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/OpenGeoAI.git
cd OpenGeoAI
pip install -r requirements.txt
```

### Quick Start Guide

1. **Set Up Your Environment**: Configure your GIS environment to use OpenGeoAI. This might involve setting up Python virtual environments and ensuring all required libraries are installed.
2. **Load Geospatial Data**: Import your geospatial datasets into OpenGeoAI using supported formats such as GeoJSON, Shapefiles, or raster data.
3. **Apply AI Models**: Use built-in AI models to analyze your geospatial data. OpenGeoAI supports various use cases, such as land cover classification, object detection, and predictive modeling.
4. **Visualize Results**: Use OpenGeoAI's visualization tools to generate maps, charts, and other visual outputs to interpret the results of your analyses.

## Use Cases

- **Land Use and Land Cover Classification**: Automatically classify land use and land cover types using satellite imagery and pre-trained AI models.
- **Urban Planning**: Analyze urban growth patterns, detect changes in infrastructure, and support smart city initiatives.
- **Environmental Monitoring**: Monitor deforestation, track wildlife movements, and assess the impact of natural disasters.
- **Agriculture**: Use AI to analyze crop health, predict yields, and optimize resource management.

## Contributing

We welcome contributions from the community! If you're interested in contributing to OpenGeoAI, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -am 'Add new feature'`).
4. Push your branch to your fork (`git push origin feature-branch`).
5. Create a pull request, and describe the changes you made.

Please make sure to follow the project's coding style and include tests for new features.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions, feedback, or collaboration inquiries, please reach out to us at [email@example.com](mailto:email@example.com).

---

Thank you for using OpenGeoAI! We hope this tool helps you unlock the full potential of AI in your geospatial projects.


