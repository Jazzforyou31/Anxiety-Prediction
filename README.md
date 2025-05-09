# Anxiety Prediction Web Application

A machine learning-powered web application that predicts anxiety levels based on user-reported health metrics. Built with Django and scikit-learn.

## Features

- **Anxiety Prediction**: Get personalized anxiety level predictions based on your health data
- **User Dashboard**: Track your anxiety levels over time with trend analysis and visualizations
- **Secure Authentication**: Create an account and securely manage your health data
- **Responsive Design**: Access the application from any device

## Demo

![Dashboard Screenshot](https://via.placeholder.com/800x400?text=Dashboard+Screenshot)

## Installation

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/anxiety-prediction-app.git
   cd anxiety-prediction-app
   ```

2. Create and activate a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Run migrations
   ```bash
   cd anxiety
   python manage.py migrate
   ```

5. Start the development server
   ```bash
   python manage.py runserver
   ```

6. Visit `http://127.0.0.1:8000/` in your browser

## Usage

1. **Register an account** or login if you already have one
2. Navigate to the **Predict** page from the dashboard
3. Fill in the form with your health metrics
4. Submit the form to receive your anxiety prediction
5. View your results and historical data on your dashboard

## Technologies Used

- **Backend**: Django 5.2
- **Frontend**: HTML, CSS, Bootstrap
- **Machine Learning**: scikit-learn (Gradient Boosting Regressor)
- **Database**: SQLite (development), PostgreSQL (recommended for production)
- **Data Processing**: pandas, numpy

## Project Structure

```
anxiety/
├── anxiety/              # Main Django project settings
├── anxiety_app/          # Main application with views, models, forms
└── manage.py             # Django command-line utility

Model and Data/
├── anxiety_model_train.py       # ML model training script
├── anxiety_prediction_model.pkl # Serialized ML model
├── test.csv                     # Test dataset
└── train.csv                    # Training dataset
```

## ML Model

The application uses a Gradient Boosting Regressor model trained on health metrics data to predict anxiety levels. The model achieves high accuracy in predicting anxiety scores based on factors such as:

- Stress-sleep relationship
- Health risk factors
- Therapy frequency
- Activity-stress balance

For more details on the model training and evaluation, see the [full project documentation](project_documentation.md).

## Deployment

For production deployment instructions, please refer to the [deployment guide](project_documentation.md#deployment-guide) in the project documentation.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 