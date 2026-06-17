"""
signal_catalog.py
=================

Danh mục các dấu hiệu nhận diện phishing dành cho LLM Agents.
Các mô tả được tối ưu hóa để AI có thể trích xuất đặc trưng (feature extraction) chính xác.
"""

SIGNAL_CATALOG = {
    "domain_spoofing": {
        "category": "url",
        "title": "Đường link có dấu hiệu giả mạo",
        "description": "Sử dụng các ký tự gần giống (homograph), sai lỗi chính tả cố ý trong tên miền hoặc sử dụng domain lạ để giả làm các dịch vụ phổ biến (ví dụ: gmaill.com, vcb-digibank.vn).",
        "severity": "high",
        "priority": 100,
    },
    "credential_request": {
        "category": "content",
        "title": "Yêu cầu thông tin nhạy cảm",
        "description": "Trực tiếp yêu cầu người dùng cung cấp mật khẩu, thông tin thẻ tín dụng, số CCCD hoặc các thông tin định danh cá nhân thông qua các biểu mẫu (form) hoặc phản hồi tin nhắn.",
        "severity": "high",
        "priority": 95,
    },
    "otp_or_code": {
        "category": "content",
        "title": "Đề cập mã OTP hoặc mã xác thực",
        "description": "Dẫn dụ hoặc yêu cầu người dùng chia sẻ mã xác thực dùng một lần (OTP) hoặc mã khôi phục tài khoản - vốn là thông tin tuyệt mật không được chia sẻ.",
        "severity": "high",
        "priority": 92,
    },
    "suspicious_attachment": {
        "category": "attachment",
        "title": "Tệp đính kèm đáng ngờ",
        "description": "Khuyến khích tải xuống hoặc mở các tệp có định dạng rủi ro (.zip, .exe, .html, .pdf) được giới thiệu là hóa đơn, tài liệu mật hoặc thông báo từ cơ quan chức năng.",
        "severity": "high",
        "priority": 90,
    },
    "extortion": {
        "category": "content",
        "title": "Có dấu hiệu tống tiền",
        "description": "Đe dọa công khai thông tin cá nhân, hình ảnh nhạy cảm hoặc lịch sử truy cập web trừ khi người dùng thanh toán bằng tiền mã hóa (crypto) hoặc chuyển khoản.",
        "severity": "critical",
        "priority": 88,
    },
    "threat": {
        "category": "content",
        "title": "Đe dọa hậu quả tiêu cực",
        "description": "Tạo ra các kịch bản tiêu cực như: khóa tài khoản vĩnh viễn, đơn kiện từ tòa án, lệnh bắt giữ hoặc bị phạt tiền nếu không thực hiện theo yêu cầu ngay lập tức.",
        "severity": "high",
        "priority": 85,
    },
    "impersonation": {
        "category": "entity",
        "title": "Có dấu hiệu mạo danh",
        "description": "Tự xưng là đại diện từ các tổ chức uy tín (Ngân hàng, CSGT, Shopee, Apple Support) nhưng sử dụng ngôn từ không chuẩn mực hoặc từ địa chỉ liên hệ không chính thức.",
        "severity": "medium",
        "priority": 80,
    },
    "financial_lure": {
        "category": "content",
        "title": "Dụ dỗ bằng lợi ích tài chính",
        "description": "Đề cập đến các khoản tiền 'từ trên trời rơi xuống' như: trúng thưởng lớn, nhận tiền hỗ trợ chính phủ, hoàn thuế, hoặc thông báo nhận quà tặng từ người lạ.",
        "severity": "medium",
        "priority": 75,
    },
    "social_engineering": {
        "category": "content",
        "title": "Thao túng tâm lý bằng sự tò mò/lo sợ",
        "description": "Sử dụng các thông tin gây sốc, tin đồn về người nổi tiếng hoặc các tình huống khẩn cấp giả tạo để kích thích người dùng bấm vào xem mà chưa kịp suy nghĩ.",
        "severity": "medium",
        "priority": 70,
    },
    "urgency": {
        "category": "content",
        "title": "Tạo áp lực thời gian",
        "description": "Sử dụng các trạng từ chỉ thời gian cực ngắn (ngay bây giờ, chỉ còn 5 phút, hạn chót hôm nay) để làm tê liệt khả năng kiểm chứng của người dùng.",
        "severity": "medium",
        "priority": 55,
    },
    "call_to_action": {
        "category": "content",
        "title": "Kêu gọi hành động rủi ro",
        "description": "Sử dụng các động từ mạnh mang tính điều hướng như: 'Click vào đây', 'Xác minh ngay', 'Đăng nhập để nhận' nhằm ép người dùng tương tác với link hoặc file.",
        "severity": "medium",
        "priority": 50,
    },
}

def get_signal_info(signal: str) -> dict:
    return SIGNAL_CATALOG.get(signal, {
        "category": "content",
        "title": f"Dấu hiệu rủi ro: {signal}",
        "description": "Hệ thống phát hiện một dấu hiệu có thể liên quan đến hành vi lừa đảo trực tuyến.",
        "severity": "medium",
        "priority": 10,
    })