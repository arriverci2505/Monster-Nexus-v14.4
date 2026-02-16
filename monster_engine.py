import os
import psutil # Nếu không có hãy pip install psutil

print("--- KIỂM TRA TIẾN TRÌNH PYTHON ---")
for proc in psutil.process_iter(['pid', 'name', 'cwd']):
    if 'python' in proc.info['name'].lower():
        print(f"PID: {proc.info['pid']} | Thư mục đang chạy: {proc.info['cwd']}")

print("\n--- TÌM FILE bot_state.json TRÊN MÁY ---")
# Lệnh này sẽ tìm file trên toàn bộ ổ đĩa (có thể hơi chậm)
