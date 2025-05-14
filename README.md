# F1 Performance Predictor 🏎️📊

## About the Project

The **F1 Performance Predictor** is an advanced machine learning system that analyzes Formula 1 racing data from 2018 to 2024 to forecast driver performance and championship standings for the 2025 season. The multiple linear regression model identifies key performance metrics and generates highly accurate projections, with an impressive **R-squared value of 0.90**, explaining 90% of the variance in driver points.

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python\&logoColor=white)](https://www.python.org/)
[![pandas](https://img.shields.io/badge/pandas-1.4.4-150458?logo=pandas\&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/numpy-1.22.4-013243?logo=numpy\&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/matplotlib-3.5.2-11557C?logo=matplotlib\&logoColor=white)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/seaborn-0.11.2-2E4C6D?logo=python\&logoColor=white)](https://seaborn.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0.2-F7931E?logo=scikit-learn\&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=Jupyter\&logoColor=white)](https://jupyter.org/)

## ✨ Key Features

* **Comprehensive Data Analysis**: 7 seasons of F1 performance data (2018–2024)
* **Predictive Modeling**: Multiple linear regression for championship points
* **2025 Projections**: Full championship standings forecast
* **Performance Insights**: Feature importance analysis
* **Consistency Metrics**: Driver performance trends across seasons

## 📚 Documentation

For complete methodology and technical details:
[Technical Whitepaper](docs/whitepaper.md) | [Model Architecture](diagrams/F1_ModelArchitecture_2018_2025.mermaid) | [Jupyter Notebook](https://github.com/binayakbartaula11/F1-Performance-Predictor/blob/main/notebooks/F1_driver_championship_prediction.ipynb)

## 📊 Dataset

**Dataset Access**:
[Formula 1 Driver Performance Dataset (2018–2024)](https://drive.google.com/drive/folders/1w_Sf8CAYqDmNBNKqdoUQHjeXmu_1BCQk)

The analysis utilizes rich historical F1 data, including:

| Feature             | Description                |
| ------------------- | -------------------------- |
| Driver Experience   | Years in F1                |
| Qualifying Position | Average grid position      |
| Race Position       | Average finishing position |
| Pit Stops           | Average per race           |
| Fastest Lap         | Average fastest lap time   |
| Championship Points | Seasonal totals            |

## 🧱 Model Architecture

```mermaid
flowchart LR
    A[📁 Historical Data<br/>2018–2024] --> B[🧹 Data Preprocessing]
    B --> C[🧬 Feature Engineering]
    C --> D[🎯 Model Training]
    D --> E[📐 Multiple Linear Regression]
    E --> F[📊 Validation<br/>R² = 0.90]
    F --> G[🔮 2025 Predictions]
    G --> H[🏁 Championship Standings]

    style A fill:#E3F2FD,stroke:#0D47A1,stroke-width:2px,color:#0D47A1
    style B fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px,color:#2E7D32
    style C fill:#FFF8E1,stroke:#F57F17,stroke-width:2px,color:#F57F17
    style D fill:#F3E5F5,stroke:#6A1B9A,stroke-width:2px,color:#6A1B9A
    style E fill:#FFEBEE,stroke:#C62828,stroke-width:2px,color:#C62828
    style F fill:#E0F7FA,stroke:#00838F,stroke-width:2px,color:#00838F
    style G fill:#F1F8E9,stroke:#558B2F,stroke-width:2px,color:#558B2F
    style H fill:#EDE7F6,stroke:#4527A0,stroke-width:2px,color:#4527A0
```

## 🔍 Key Findings

| Factor        | Impact              | Coefficient |
| ------------- | ------------------- | ----------- |
| Race Position | Strongest predictor | -48.94      |
| Pit Stops     | Strategic advantage | +291.28     |
| Experience    | Performance boost   | +4.88       |
| Lap Times     | Speed matters       | +1.80       |
| Qualifying    | Grid importance     | -1.13       |

**Model Accuracy**: 90% variance explained (R²=0.90)

## 🏆 2025 Projected Standings

| Rank | Driver          | Points | Trend |
| ---- | --------------- | ------ | ----- |
| 1    | Max Verstappen  | 405    | ▲     |
| 2    | Lando Norris    | 313    | ▲▲    |
| 3    | Charles Leclerc | 300    | ▲     |
| 4    | Lewis Hamilton  | 224    | ▼     |
| 5    | Sergio Perez    | 210    | ▼     |
| 6    | Oscar Piastri   | 191    | ▲▲    |
| 7    | Valtteri Bottas | 0      | ▼▼    |

### Legend:

* **▲▲**: Significant projected improvement
* **▲**: Moderate projected improvement
* **▼**: Moderate projected decline
* **▼▼**: Significant projected decline

## 🚀 Getting Started

### Prerequisites

* Python 3.8+
* Data Science Stack (pandas, numpy, scikit-learn)
* Visualization Tools (matplotlib, seaborn)

### Installation

```bash
# Clone repository
git clone https://github.com/binayakbartaula11/F1-Performance-Predictor.git
cd F1-Performance-Predictor

# Install dependencies
pip install -r requirements.txt
```

### Usage

```bash
# Run full analysis
python f1_analysis.py

# Generate predictions only
python predict_2025.py
```

## 📂 Data Structure

```
f1_performance_2018_2024.csv
├── Year
├── Driver
├── Driver_Experience
├── Avg_Qualifying_Position
├── Avg_Race_Position
├── Pit_Stops
├── Fastest_Lap_Times
└── Driver_Points
```

### 🏗️  **Final Thoughts**

The model’s projections remain within a reasonable margin of plausibility, particularly in light of the current performance trends exhibited by emerging drivers such as Oscar Piastri and Lando Norris. However, Formula 1 is characterized by a highly dynamic environment in which team developments, regulatory adjustments, and mid-season upgrades can substantially alter the competitive landscape.

As the 2025 season progresses, especially through the European and Asian rounds, ongoing monitoring of performance indicators will be essential to refining expectations regarding potential championship outcomes.

> 🏁 The 2025 Formula 1 season is scheduled to conclude with the **Abu Dhabi Grand Prix on December 7, 2025**. Until that time, the championship standings remain subject to significant change.

---

## 🤝 Contributing

Contributions are welcome. To contribute:

1. Fork the repository
2. Create a new feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

## 📜 License

MIT Licensed - See [LICENSE](LICENSE) for details.

## 🌟 Acknowledgments

* Formula 1 for the incredible sport
* F1 data community for maintaining datasets
* Python data science contributors

## 📝 Citation

```bibtex
@misc{F1PP2025,
  author = {Binayak Bartaula},
  title = {F1 Performance Predictor},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/binayakbartaula11/F1-Performance-Predictor}
}
```
