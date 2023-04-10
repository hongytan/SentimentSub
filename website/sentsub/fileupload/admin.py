from django.contrib import admin
from .models import Video, CaptionVideo

# Register your models here.
admin.site.register(Video) # Register Post model onto admin page
admin.site.register(CaptionVideo)