# Student Grade Prediction System

A comprehensive web-based machine learning system for predicting student final grades based on academic performance, study habits, and demographic factors. The system uses a 40% pass mark threshold and provides personalized recommendations for academic improvement.

## 📸 Screenshots

<img width="516" height="749" alt="Screenshot From 2025-07-21 22-54-27" src="https://github.com/user-attachments/assets/9248beb3-f343-42af-9067-fe2a894c6f1f" />


## 🎯 Features

- **Accurate Grade Prediction**: ML-powered predictions using Random Forest and Gradient Boosting algorithms
- **40% Pass Mark System**: Aligned with standard university grading where 40% is the minimum passing grade
- **Pass/Fail Status**: Clear indication of whether student meets the pass threshold
- **Personalized Recommendations**: Tailored advice based on predicted performance and risk level
- **Real-time Analysis**: Instant predictions with confidence scores and probability distributions
- **Comprehensive Scoring**: Multi-factor analysis including academic (60%), study habits (20%), resources (15%), and background (5%)
- **Risk Assessment**: Early identification of at-risk students requiring intervention

## 🏗️ System Architecture

### Frontend
- **Next.js 14** with App Router
- **TypeScript** for type safety
- **Tailwind CSS** for styling
- **shadcn/ui** components for modern UI
- **Responsive Design** for all devices

### Backend
- **Next.js API Routes** for prediction endpoints
- **Python Scripts** for model training and analysis
- **scikit-learn** for machine learning algorithms
- **pandas & numpy** for data processing

### Machine Learning
- **Random Forest Classifier** for grade prediction
- **Gradient Boosting** for enhanced accuracy
- **Feature Engineering** with 15+ derived features
- **Cross-validation** for model reliability
- **Class Balancing** for fair predictions across all grades

## 📋 Prerequisites

Before installation, ensure you have:

- **Node.js** 18.0 or higher
- **npm** or **yarn** package manager
- **Python** 3.8 or higher (for model training)
- **Git** for version control

## 🚀 Installation Guide

### Method 1: Quick Installation (Recommended)

1. **Download the project** using v0's built-in installation:
   - Click the "Download Code" button in the top-right corner of the Block view
   - Choose "Create New Project" or "Add to Existing Project"
   - Follow the shadcn CLI setup prompts

2. **Navigate to project directory**:
   \`\`\`bash
   cd student-grade-prediction
   \`\`\`

3. **Install dependencies**:
   \`\`\`bash
   npm install
   \`\`\`

4. **Start the development server**:
   \`\`\`bash
   npm run dev
   \`\`\`

5. **Open your browser** and visit \`http://localhost:3000\`

### Method 2: Manual Installation

1. **Clone or download the repository**:
   \`\`\`bash
   git clone <repository-url>
   cd student-grade-prediction
   \`\`\`

2. **Install Node.js dependencies**:
   \`\`\`bash
   npm install
   # or
   yarn install
   \`\`\`

3. **Set up Python environment** (optional, for model training):
   \`\`\`bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   pip install pandas numpy scikit-learn matplotlib seaborn
   \`\`\`

4. **Start the development server**:
   \`\`\`bash
   npm run dev
   \`\`\`



## 🔧 Configuration

### Environment Variables

Create a \`.env.local\` file in the root directory for any environment-specific configurations:

\`\`\`env
# Add any API keys or configuration here
# Currently, no external APIs are required
\`\`\`

### Model Configuration

The prediction algorithm can be customized in \`app/api/predict/route.ts\`:

- **Grade Boundaries**: Adjust percentage thresholds for each grade
- **Weightings**: Modify the importance of different factors
- **Pass Mark**: Change the 40% pass threshold if needed
- **Confidence Levels**: Adjust prediction confidence calculations

## 🎮 Usage Guide

### Basic Usage

1. **Open the application** in your web browser
2. **Fill in student information**:
   - Demographics (age, gender)
   - Family background (income, parent education)
   - Study habits (hours per week, library visits, online usage)
   - Academic scores (CT1, CT2, assignments, presentations, midterm, final)
3. **Click "Predict Grade"** to get results
4. **Review the prediction**:
   - Pass/Fail status
   - Predicted grade and confidence
   - Grade probabilities
   - Score breakdown
   - Personalized recommendations

### Input Guidelines

#### Academic Scores
- **CT1 & CT2**: Out of 10 points each
- **Assignment**: Out of 20 points
- **Presentation**: Out of 15 points
- **Midterm**: Out of 30 points
- **Final Exam**: Out of 40 points

#### Study Habits
- **Study Hours**: Weekly hours dedicated to studying
- **Library Visits**: Monthly visits to library facilities
- **Online Resources**: Frequency of using online educational materials

### Understanding Results

#### Pass/Fail Status
- **PASS**: Student meets the 40% minimum requirement
- **FAIL**: Student falls below the 40% pass mark

#### Grade Scale
- **A+ (80%+)**: Excellent performance
- **A (75-80%)**: Very good performance
- **A- (70-75%)**: Good performance
- **B+ (65-70%)**: Above average
- **B (60-65%)**: Average performance
- **B- (55-60%)**: Below average but satisfactory
- **C+ (50-55%)**: Satisfactory
- **C (45-50%)**: Acceptable
- **C- (40-45%)**: Minimum passing grade
- **D (35-40%)**: Below pass mark
- **F (0-35%)**: Clear failure

## 🧪 Model Training & Validation

### Running Model Training Scripts

1. **Activate Python environment**:
   \`\`\`bash
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   \`\`\`

2. **Run data analysis**:
   \`\`\`bash
   python scripts/advanced_model_development.py
   \`\`\`

3. **Validate model performance**:
   \`\`\`bash
   python scripts/model_validation.py
   \`\`\`

4. **Analyze pass mark system**:
   \`\`\`bash
   python scripts/pass_mark_model_analysis.py
   \`\`\`

### Model Performance Metrics

- **Balanced Accuracy**: ~75-85% (varies by dataset)
- **Pass/Fail Accuracy**: ~80-90%
- **Cross-validation**: 5-fold stratified validation
- **Feature Importance**: Academic scores (highest), study habits, resource usage

## 🛠️ Troubleshooting

### Common Issues

#### Installation Problems
- **Node.js version**: Ensure you're using Node.js 18+
- **Package conflicts**: Delete \`node_modules\` and \`package-lock.json\`, then reinstall
- **Python dependencies**: Use a virtual environment for Python packages

#### Runtime Errors
- **API errors**: Check browser console for detailed error messages
- **Prediction failures**: Ensure all required fields are filled
- **Performance issues**: Check if all dependencies are properly installed

#### Model Issues
- **Inaccurate predictions**: Retrain model with more recent data
- **Bias in results**: Check class balancing in training scripts
- **Low confidence**: Review feature engineering and model parameters

### Getting Help

1. **Check the browser console** for error messages
2. **Verify input data** meets the expected format
3. **Review the API response** for error details
4. **Check GitHub issues** for known problems
5. **Contact support** through the appropriate channels

## 📊 Model Details

### Feature Engineering

The system creates 15+ engineered features from raw inputs:

- **CT Analysis**: Improvement trends, consistency, averages
- **Academic Performance**: Weighted scores, standard deviation, ranges
- **Study Behavior**: Intensity categories, efficiency metrics
- **Resource Utilization**: Combined library and online usage scores
- **Socioeconomic Factors**: Income and education impact scores

### Algorithm Selection

- **Primary**: Random Forest Classifier with balanced class weights
- **Secondary**: Gradient Boosting for comparison
- **Selection**: Best performing model based on cross-validation
- **Optimization**: Grid search for hyperparameter tuning

### Validation Methods

- **Stratified K-Fold**: Maintains grade distribution in splits
- **Balanced Accuracy**: Accounts for class imbalance
- **Confusion Matrix**: Detailed per-class performance analysis
- **Feature Importance**: Identifies most predictive factors

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

## 🙏 Acknowledgments

- **Dataset**: Green University Student Grade Prediction Dataset
- **Framework**: Next.js and React ecosystem
- **ML Libraries**: scikit-learn, pandas, numpy
- **UI Components**: shadcn/ui component library
- **Styling**: Tailwind CSS framework


---

**Built with ❤️ using Next.js, TypeScript, and Machine Learning**
