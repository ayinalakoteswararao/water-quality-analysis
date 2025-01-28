import warnings
# Filter out specific scikit-learn warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from flask import Flask, render_template, request, jsonify, url_for
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
import joblib
from datetime import datetime
from flask_mail import Mail, Message
import random

app = Flask(__name__)

# Static file configuration
app.config['STATIC_FOLDER'] = 'static'

# Email configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'ayinalakoteswararao@gmail.com'
app.config['MAIL_PASSWORD'] = 'kmmq qywm anry tsre'
app.config['MAIL_DEFAULT_SENDER'] = 'ayinalakoteswararao@gmail.com'

mail = Mail(app)

# Sample data for the about page
members = [
    {"name": "Member 1", "role": "Role 1"}
]

blog_content = [
    {
        "icon": "fa-tint",
        "title": "Water Quality Basics",
        "content": "Understanding water quality is crucial for public health and environmental protection. Clean water is essential for drinking, agriculture, and ecosystem balance."
    },
    {
        "icon": "fa-flask",
        "title": "Testing Methods",
        "content": "Modern water quality testing involves various parameters including pH levels, dissolved oxygen, turbidity, and chemical composition analysis."
    },
    {
        "icon": "fa-leaf",
        "title": "Environmental Impact",
        "content": "Water quality directly affects ecosystems, marine life, and biodiversity. Maintaining clean water sources is vital for environmental sustainability."
    },
    {
        "icon": "fa-home",
        "title": "Household Water Safety",
        "content": "Learn about maintaining water quality in your home, including filtration systems, regular testing, and best practices for water storage."
    },
    {
        "icon": "fa-microscope",
        "title": "Advanced Analysis",
        "content": "Discover the latest technologies and methods used in water quality analysis, from molecular testing to real-time monitoring systems."
    },
    {
        "icon": "fa-industry",
        "title": "Industrial Management",
        "content": "Explore how industries maintain water quality standards and implement sustainable water management practices."
    },
    {
        "icon": "fa-shower",
        "title": "Daily Water Usage",
        "content": "Tips and insights on how to optimize your daily water usage while maintaining quality and safety standards."
    },
    {
        "icon": "fa-filter",
        "title": "Filtration Systems",
        "content": "Compare different water filtration systems and learn which one best suits your needs for clean, safe water."
    },
    {
        "icon": "fa-seedling",
        "title": "Agricultural Impact",
        "content": "How water quality affects crop growth, soil health, and sustainable farming practices in modern agriculture."
    },
    {
        "icon": "fa-fish",
        "title": "Aquatic Ecosystems",
        "content": "Understanding the delicate balance of aquatic ecosystems and how water quality influences marine life."
    },
    {
        "icon": "fa-cloud-rain",
        "title": "Rainwater Harvesting",
        "content": "Learn about collecting and storing rainwater safely for various purposes while maintaining quality standards."
    },
    {
        "icon": "fa-temperature-high",
        "title": "Temperature Effects",
        "content": "How temperature changes affect water quality and the importance of monitoring thermal pollution."
    },
    {
        "icon": "fa-city",
        "title": "Urban Water Systems",
        "content": "Exploring city water management, treatment facilities, and distribution networks for clean water."
    },
    {
        "icon": "fa-hand-holding-water",
        "title": "Conservation Tips",
        "content": "Practical ways to conserve water in daily life while maintaining its quality for future generations."
    },
    {
        "icon": "fa-vial",
        "title": "Chemical Analysis",
        "content": "Understanding the chemical parameters that determine water quality and their significance."
    },
    {
        "icon": "fa-bacteria",
        "title": "Microbial Safety",
        "content": "Learn about waterborne pathogens, testing methods, and ensuring microbiological safety of water."
    }
]

# Load and preprocess the dataset
def load_and_preprocess_data():
    data = pd.read_csv('water_potability.csv')
    data.dropna(inplace=True)
    X = data.drop('Potability', axis=1)
    y = data['Potability']
    
    # Convert data to float32 and handle missing values
    X = X.astype('float32')
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.mean(), inplace=True)
    
    return X, y

# Load the model
def load_model():
    model_path = "voting_classifier_model.pkl"
    return joblib.load(model_path)

# Initialize data and model
X, y = load_and_preprocess_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = load_model()

# Evaluate model accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

def predict_potability(features):
    try:
        input_data = np.array(features).reshape(1, -1)
        prediction = model.predict(input_data)
        proba = model.predict_proba(input_data)[0]
        return prediction[0], max(proba)
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return None, None

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get user input from the form
            ph = float(request.form['ph'])
            hardness = float(request.form['Hardness'])
            solids = float(request.form['Solids'])
            chloramines = float(request.form['Chloramines'])
            sulfate = float(request.form['Sulfate'])
            conductivity = float(request.form['Conductivity'])
            organic_carbon = float(request.form['Organic_carbon'])
            trihalomethanes = float(request.form['Trihalomethanes'])
            turbidity = float(request.form['Turbidity'])

            # Validate input ranges
            if not (0 <= ph <= 14):
                return render_template('index.html', error="pH must be between 0 and 14")
            
            if any(x < 0 for x in [hardness, solids, chloramines, sulfate, conductivity, 
                                  organic_carbon, trihalomethanes, turbidity]):
                return render_template('index.html', error="All values must be positive")

            # Create feature list
            features = [ph, hardness, solids, chloramines, sulfate, conductivity, 
                       organic_carbon, trihalomethanes, turbidity]
            
            # Get prediction and confidence
            prediction, confidence = predict_potability(features)
            
            if prediction is None:
                return render_template('index.html', error="Error making prediction")
            
            # Convert prediction to human-readable result
            result = "Potable" if prediction == 1 else "Not Potable"
            
            return render_template('result.html', 
                                 prediction=result,
                                 confidence=round(confidence * 100, 2),
                                 ph=ph,
                                 hardness=hardness,
                                 solids=solids,
                                 chloramines=chloramines,
                                 sulfate=sulfate,
                                 conductivity=conductivity,
                                 organic_carbon=organic_carbon,
                                 trihalomethanes=trihalomethanes,
                                 turbidity=turbidity)
            
        except ValueError:
            return render_template('index.html', error="Please enter valid numerical values")
        except Exception as e:
            return render_template('index.html', error=f"Error: {str(e)}")

@app.route('/blog')
def blog():
    # Randomly select 8 blog posts without repetition
    selected_posts = random.sample(blog_content, 8)
    return render_template('blog.html', posts=selected_posts)

@app.route('/about')
def about():
    team_member = {
        "name": "Ayinala Koteswara Rao",
        "role": "Lead Developer & Water Quality Expert",
        "image": url_for('static', filename='images/member1.jpg'),
        "description": "Lead Developer & Water Quality Expert with extensive experience in environmental engineering. Specializing in developing innovative solutions for water quality analysis and monitoring systems.",
        "education": "B.Tech in Artificial Intelligence & Machine Learning",
        "expertise": [
            "Full Stack Development",
            "Machine Learning",
        ],
        "social_links": {
            "linkedin": "https://www.linkedin.com/in/ayinala-koteswararao-711bab271/",
            "github": "https://github.com/ayinalakoteswararao",
            "email": "ayinalakoteswararao@gmail.com"
        }
    }
    return render_template('about.html', member=team_member)

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        phone = request.form.get('phone')
        message = request.form.get('message')
        
        # Create email content with HTML formatting
        msg_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #ddd; border-radius: 5px;">
                <h2 style="color: #2c3e50; text-align: center; border-bottom: 2px solid #3498db; padding-bottom: 10px;">
                    📬 New Contact Form Submission
                </h2>
                
                <div style="margin: 20px 0;">
                    <p style="margin: 10px 0;">
                        <strong>👤 Name:</strong> {name}
                    </p>
                    <p style="margin: 10px 0;">
                        <strong>📧 Email:</strong> {email}
                    </p>
                    <p style="margin: 10px 0;">
                        <strong>📱 Phone:</strong> {phone}
                    </p>
                </div>
                
                <div style="background-color: #f9f9f9; padding: 15px; border-radius: 5px;">
                    <h3 style="color: #2c3e50; margin-top: 0;">💬 Message:</h3>
                    <p style="white-space: pre-line;">{message}</p>
                </div>
                
                <div style="margin-top: 20px; text-align: center; color: #666; font-size: 12px;">
                    <p>This is an automated message from your Water Quality Analysis System</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        try:
            msg = Message(
                subject='New Contact Form Submission',
                recipients=['ayinalakoteswararao@gmail.com'],
                html=msg_body  # Changed from body to html to support HTML formatting
            )
            mail.send(msg)
            return jsonify({'success': True, 'message': 'Message sent successfully!'})
        except Exception as e:
            print(f"Error sending email: {e}")
            return jsonify({'success': False, 'message': 'Failed to send message. Please try again later.'}), 500
            
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
