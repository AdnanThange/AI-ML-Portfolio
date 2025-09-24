from django.db import models

class HeartPrediction(models.Model):
    # User input fields
    age = models.PositiveIntegerField()
    sex = models.CharField(max_length=1)  # M or F
    chest_pain_type = models.CharField(max_length=10)
    resting_bp = models.FloatField()
    cholesterol = models.FloatField()
    fasting_bs = models.PositiveSmallIntegerField()  # 0 or 1
    resting_ecg = models.CharField(max_length=10)
    max_hr = models.FloatField()
    exercise_angina = models.CharField(max_length=1)  # Y or N
    oldpeak = models.FloatField()
    st_slope = models.CharField(max_length=10)

    # Prediction result
    prediction = models.CharField(max_length=20)  # "Disease" or "No Disease"

    # Timestamp
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.age}y, {self.sex}, Prediction: {self.prediction}"
