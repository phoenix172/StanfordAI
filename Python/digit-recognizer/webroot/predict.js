function predict_clicked() {
    $.post('predict',
    document.getElementById('myCanvas').toDataURL("image/png"),
    function (data) {
        alert(data);
    });
}