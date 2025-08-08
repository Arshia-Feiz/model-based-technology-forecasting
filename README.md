# Model-Based Approaches for Technology Forecasting and Game-Theoretic Modelling

## 📊 Project Overview

This project recreates the analytical pipeline from the research paper "Model-based Approaches for Technology Planning Roadmapping: Technology Forecasting and Game-theoretic Modelling" by Golkar et al. The work focuses on analyzing competitive dynamics in the automotive industry using game-theoretic models and technology forecasting techniques.

## 🎯 Key Objectives

- **Research Reproduction**: Recreate analytical methodology from academic literature
- **Competitive Analysis**: Analyze manufacturer competition in automotive industry
- **Technology Trends**: Identify performance trade-offs and optimization strategies
- **Game Theory Application**: Model strategic interactions between manufacturers

## 🛠️ Technical Stack

- **Python**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Matplotlib/Seaborn**: Data visualization
- **Scikit-learn**: Statistical analysis
- **Jupyter Notebooks**: Interactive analysis and documentation

## 📈 Methodology

### 1. Data Substitution and Preprocessing
- **Original Dataset**: Academic scraped data (substituted with public Kaggle dataset)
- **Manufacturers**: BMW, Audi, Mercedes-Benz
- **Metrics**: Engine power, fuel consumption, acceleration
- **Cleaning**: Removed non-numeric values, outliers, and irrelevant rows

### 2. Exploratory Data Analysis
- **Performance Visualization**: Horsepower vs. acceleration plots
- **Efficiency Analysis**: Fuel consumption vs. horsepower relationships
- **Temporal Trends**: Performance evolution over time
- **Manufacturer Comparison**: Competitive positioning analysis

### 3. Best Response Analysis
- **Yearly Optimization**: Maximum horsepower and minimum acceleration per year
- **Competitive Dynamics**: How companies respond to each other's strategies
- **Technology Progression**: Validation of competitive assumptions
- **Strategic Positioning**: Understanding manufacturer optimization strategies

## 📊 Results

### Key Visualizations
- **Performance Trade-offs**: Horsepower vs. acceleration scatter plots
- **Efficiency Metrics**: Fuel consumption vs. performance relationships
- **Temporal Evolution**: Performance trends over time
- **Competitive Landscape**: Manufacturer positioning analysis

### Strategic Insights
- **Optimization Patterns**: How manufacturers maximize/minimize specific metrics
- **Competitive Responses**: Strategic reactions to competitor actions
- **Technology Evolution**: Long-term performance improvement trends
- **Market Dynamics**: Understanding competitive pressures

## 🚀 Usage

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Data Requirements
- Kaggle car specifications dataset
- BMW, Audi, Mercedes-Benz vehicle data
- Performance metrics: horsepower, acceleration, fuel consumption
- Temporal data: Year of manufacture

### Analysis Workflow
```python
# Example analysis setup
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess data
df = pd.read_csv('car_specifications.csv')
manufacturers = ['BMW', 'Audi', 'Mercedes-Benz']
filtered_df = df[df['Manufacturer'].isin(manufacturers)]

# Create performance visualizations
plt.figure(figsize=(12, 8))
sns.scatterplot(data=filtered_df, x='Horsepower', y='Acceleration', 
                hue='Manufacturer', style='Year')
```

## 📁 Project Structure

```
model-based-technology-forecasting/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   └── car_specifications.csv
│   └── processed/
│       └── filtered_manufacturers.csv
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_competitive_analysis.ipynb
│   └── 04_game_theory_validation.ipynb
├── src/
│   ├── data_processing.py
│   ├── visualization.py
│   ├── competitive_analysis.py
│   └── game_theory_models.py
├── results/
│   ├── visualizations/
│   │   ├── performance_tradeoffs.png
│   │   ├── efficiency_analysis.png
│   │   └── competitive_landscape.png
│   └── analysis/
│       └── best_response_analysis.csv
└── documentation/
    └── methodology_notes.md
```

## 🔬 Research Applications

This project demonstrates:
- **Academic Research Reproduction**: Validating published methodologies
- **Competitive Intelligence**: Understanding industry dynamics
- **Technology Strategy**: Supporting strategic decision-making
- **Game Theory Application**: Modeling competitive interactions

## 📚 References

- **Original Paper**: Golkar et al. "Model-based Approaches for Technology Planning Roadmapping"
- **Dataset**: [Kaggle Car Specifications](https://www.kaggle.com/datasets/CooperUnion/car-dataset)
- **Game Theory**: Strategic interaction modeling in technology markets

## 🎓 Academic Context

This work was conducted at ISAE-SUPAERO as part of research into:
- Technology forecasting methodologies
- Competitive analysis in automotive industry
- Game-theoretic modeling applications
- Strategic technology planning

## 👨‍💻 Author

**Arshia Feizmohammady**
- Industrial Engineering Student, University of Toronto
- Research focus: Technology forecasting and competitive analysis
- [LinkedIn](https://linkedin.com/in/arshiafeiz)
- [Personal Website](https://arshiafeizmohammady.com)

## 📄 License

This project is for educational and research purposes. Please cite the original paper and this reproduction appropriately.

---

*This project reproduces academic research methodology for educational purposes and demonstrates the application of game theory in technology forecasting.*
