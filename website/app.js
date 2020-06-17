$(function() {
    $('#submit').click(function() {
        event.preventDefault();
        var form_data = new FormData($('#uploadform')[0]);
        $.ajax({
            type: 'POST',
            url: 'http://9e3622b716b1.ngrok.io/upload',
            data: form_data,
            contentType: false,
            processData: false,
            dataType: 'json',
        }).done(function(data){
            console.log(data);
            console.log('Success!');
        }).fail(function(data){
            alert('error!');
        });
    });
}); 