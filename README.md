# Data Analysis Dashboard with Streamlit

## Project Overview
This project is a data analysis dashboard built using **Streamlit** to visualize and perform descriptive statistics on a given dataset. The application is designed to allow users to upload datasets, clean the data, and generate visual insights such as correlation matrices, histograms, pie charts, and more. The project is tailored for general-purpose descriptive analysis, providing both numerical and categorical data insights.

## Features
- **Data Cleaning**: Automatically handles missing values by dropping empty rows and resetting indices.
- **Descriptive Statistics**: Presents detailed statistical summaries including count, mean, standard deviation, min, max, and percentiles for numerical columns.
- **Correlation Heatmap**: Visualizes relationships between numerical variables through a correlation matrix.
- **Histograms with KDE**: Displays the distribution of numerical columns using histograms with optional KDE (Kernel Density Estimate) for smooth curves.
- **Categorical Analysis**: Generates pie charts for categorical columns to visualize distribution.
- **Outlier Detection**: Uses box plots to detect outliers in numerical data.

## Technologies Used
- **Python**: Core programming language.
- **Streamlit**: For building the interactive data analysis dashboard.
- **Pandas**: For data manipulation and cleaning.
- **Seaborn & Matplotlib**: For advanced data visualization.
- **NumPy**: For numerical operations.

## How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/data-analysis-dashboard.git
   cd data-analysis-dashboard
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

4. Upload your dataset and explore the interactive visualizations!

## Project Structure
```
├── app.py                  # Main Streamlit application script
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
└── data/                    # Folder for storing datasets (optional)
```
