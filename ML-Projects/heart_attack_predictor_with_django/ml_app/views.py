from django.shortcuts import render
from .heart_model import predict_heart_disease
from .models import HeartPrediction

# Safe allowed values for dropdowns
SEX_CHOICES = ["M", "F"]
CHEST_PAIN_TYPES = ["ATA", "NAP", "ASY", "TA"]
RESTING_ECG_CHOICES = ["Normal", "ST", "LVH"]
EXERCISE_ANGINA_CHOICES = ["Y", "N"]
ST_SLOPE_CHOICES = ["Up", "Flat", "Down"]

def home(request):
    prediction = None
    last_predictions = HeartPrediction.objects.all().order_by('-created_at')[:5]

    if request.method == "POST":
        # Collect and sanitize form data
        input_data = {
            "Age": int(request.POST.get("Age")),
            "Sex": request.POST.get("Sex").strip(),
            "ChestPainType": request.POST.get("ChestPainType").strip(),
            "RestingBP": float(request.POST.get("RestingBP")),
            "Cholesterol": float(request.POST.get("Cholesterol")),
            "FastingBS": int(request.POST.get("FastingBS")),
            "RestingECG": request.POST.get("RestingECG").strip(),
            "MaxHR": float(request.POST.get("MaxHR")),
            "ExerciseAngina": request.POST.get("ExerciseAngina").strip(),
            "Oldpeak": float(request.POST.get("Oldpeak")),
            "ST_Slope": request.POST.get("ST_Slope").strip()
        }

        # Ensure values are valid
        if input_data["Sex"] not in SEX_CHOICES or \
           input_data["ChestPainType"] not in CHEST_PAIN_TYPES or \
           input_data["RestingECG"] not in RESTING_ECG_CHOICES or \
           input_data["ExerciseAngina"] not in EXERCISE_ANGINA_CHOICES or \
           input_data["ST_Slope"] not in ST_SLOPE_CHOICES:
            prediction = "Invalid input!"
        else:
            # Get prediction
            prediction = predict_heart_disease(input_data)

            # Save prediction to database
            HeartPrediction.objects.create(
                age=input_data["Age"],
                sex=input_data["Sex"],
                chest_pain_type=input_data["ChestPainType"],
                resting_bp=input_data["RestingBP"],
                cholesterol=input_data["Cholesterol"],
                fasting_bs=input_data["FastingBS"],
                resting_ecg=input_data["RestingECG"],
                max_hr=input_data["MaxHR"],
                exercise_angina=input_data["ExerciseAngina"],
                oldpeak=input_data["Oldpeak"],
                st_slope=input_data["ST_Slope"],
                prediction=prediction
            )

            # Refresh last predictions
            last_predictions = HeartPrediction.objects.all().order_by('-created_at')[:5]

    context = {
        "prediction": prediction,
        "last_predictions": last_predictions,
        "sex_choices": SEX_CHOICES,
        "chest_pain_types": CHEST_PAIN_TYPES,
        "resting_ecg_choices": RESTING_ECG_CHOICES,
        "exercise_angina_choices": EXERCISE_ANGINA_CHOICES,
        "st_slope_choices": ST_SLOPE_CHOICES
    }
    return render(request, 'home.html', context)
