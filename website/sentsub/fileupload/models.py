from django.db import models

class Video(models.Model):
    name= models.CharField(max_length=500)
    videofile= models.FileField(upload_to='videos/', null=True,verbose_name="")

    def __str__(self):
        return self.name + ": " + str(self.videofile)

class CaptionVideo(models.Model):
    cvideo = models.FileField(upload_to='captions/', null=True)

    def __str__(self):
        return self.name + ': ' + str(self.cvideo)