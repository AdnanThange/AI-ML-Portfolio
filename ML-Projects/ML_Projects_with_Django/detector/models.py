from django.db import models

from django.db import models

# Keep minimal. If you'd like to store detection results, add models here.
class UploadedImage(models.Model):
    image = models.ImageField(upload_to="uploads/")
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"UploadedImage {self.id}"

