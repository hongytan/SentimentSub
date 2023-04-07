document.getElementById('uploadForm').addEventListener('submit', function(event) {
    event.preventDefault();
    var fileInput = document.getElementById('fileInput');
    var statusElement = document.getElementById('status');
    statusElement.textContent = 'Uploading...';
    
    var formData = new FormData();
    formData.append('mp4File', fileInput.files[0]);
    
    var xhr = new XMLHttpRequest();
    xhr.open('POST', 'upload.php', true);
    xhr.onreadystatechange = function() {
        if (xhr.readyState === XMLHttpRequest.DONE) {
            if (xhr.status === 200) {
                statusElement.textContent = 'Upload complete!';
            } else {
                statusElement.textContent = 'Upload failed. Please try again.';
            }
        }
    };
    xhr.send(formData);
});
