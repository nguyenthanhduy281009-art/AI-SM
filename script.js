function generateImage() {
    const prompt = document.getElementById('prompt').value;
    const resultDiv = document.getElementById('result');
    resultDiv.innerHTML = "Đang tạo ảnh, vui lòng chờ...";
    
    // Đây là nơi bạn sẽ gọi API sau này
    console.log("Đang gửi yêu cầu:", prompt);
}
