from django.shortcuts import render
from django.http import StreamingHttpResponse
from django.conf import settings
import os

from .forms import UploadImageForm
from .yolo_detect import detect_objects  # for uploaded images
from .stream import gen  # live camera generator

# ------------------------
# Home Page
# ------------------------
def home(request):
    return render(request, "home.html")

# ------------------------
# Image Upload & Detection
# ------------------------
def upload_image(request):
    detected_image_url = None  # URL for template

    if request.method == "POST":
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data["image"]

            # Save uploaded image in MEDIA_ROOT
            upload_path = os.path.join(settings.MEDIA_ROOT, image.name)
            with open(upload_path, "wb+") as f:
                for chunk in image.chunks():
                    f.write(chunk)

            # Run YOLO detection
            output_path = detect_objects(upload_path)

            # Build URL for template
            detected_image_url = settings.MEDIA_URL + os.path.basename(output_path)

    else:
        form = UploadImageForm()

    return render(
        request,
        "upload.html",
        {"form": form, "detected_image_path": detected_image_url}
    )

# ------------------------
# Live Camera Page
# ------------------------
def live_page(request):
    """Render the live camera page"""
    return render(request, "live.html")

# ------------------------
# Live Camera Feed (MJPEG Streaming)
# ------------------------
def live_camera_feed(request):
    """Stream frames from camera with YOLO detection"""
    return StreamingHttpResponse(
        gen(),
        content_type="multipart/x-mixed-replace; boundary=frame"
    )
