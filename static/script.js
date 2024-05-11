$(document).ready(function() {
    $('#input-form').submit(function(event) {
        event.preventDefault();
        var form = $(this);
        var url = form.attr('action');
        var formData = new FormData(form[0]);

        $.ajax({
            type: 'POST',
            url: url,
            data: formData,
            processData: false,
            contentType: false,
            success: function(response) {
                var resultDiv = $('#result');
                resultDiv.empty();

                var sentimentScore = response.sentiment_score;
                var faceDetected = response.face_detected;
                var expressions = response.expressions;

                resultDiv.append('<p>Sentiment Score: ' + JSON.stringify(sentimentScore) + '</p>');
                resultDiv.append('<p>Face Detected: ' + faceDetected + '</p>');
                resultDiv.append('<p>Expressions: ' + JSON.stringify(expressions) + '</p>');
            }
        });
    });
});