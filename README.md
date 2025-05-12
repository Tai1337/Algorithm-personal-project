# Project Giải Đố 8-Puzzle và Game Mê Cung Thông Minh

Chào mừng bạn đến với dự án kết hợp trình giải đố 8-Puzzle mạnh mẽ với nhiều thuật toán tìm kiếm khác nhau và một trò chơi mê cung (Chuột tìm Phô mai) được điều khiển bởi AI sử dụng Q-Learning!

Giao diện chính (`GiaoDien.py`) được xây dựng bằng CustomTkinter, cho phép bạn:
* Thiết lập trạng thái bắt đầu và trạng thái đích cho bài toán 8-Puzzle.
* Chọn từ một danh sách đa dạng các thuật toán tìm kiếm để giải đố.
* Theo dõi trực quan quá trình giải đố từng bước.
* Xem thống kê về hiệu suất của các thuật toán.
* Khởi chạy và xem agent Q-Learning chơi trò chơi Chuột tìm Phô mai.

## Tổng Quan Các Thuật Toán

Dự án này triển khai và cho phép bạn khám phá một loạt các thuật toán tìm kiếm, được phân loại như sau:

**Nhóm 1: Tìm kiếm không có thông tin (Uninformed Search)**
* Breadth-First Search (BFS) - Tìm kiếm theo chiều rộng
* Depth-First Search (DFS) - Tìm kiếm theo chiều sâu
* Uniform Cost Search (UCS) - Tìm kiếm chi phí đồng nhất
* Iterative Deepening DFS (IDDFS) - Tìm kiếm sâu dần lặp (Kết hợp DFS + Depth-Limited Search)

**Nhóm 2: Tìm kiếm có thông tin (Informed Search / Heuristic Search)**
* Greedy Best-First Search - Tìm kiếm tham lam tốt nhất đầu tiên
* A* Search (A Sao)
* Iterative Deepening A* (IDA*) - A* sâu dần lặp (Kết hợp A* + IDDFS)

**Nhóm 3: Tìm kiếm cục bộ (Local Search)**
* Hill Climbing - Leo đồi:
    * Simple Hill Climbing - Leo đồi đơn giản
    * Steepest-Ascent Hill Climbing - Leo đồi dốc nhất
    * Stochastic Hill Climbing - Leo đồi ngẫu nhiên
* Simulated Annealing - Ủ mô phỏng
* Genetic Algorithms - Thuật toán di truyền
* Beam Search - Tìm kiếm chùm tia

**Nhóm 4: Tìm kiếm trong môi trường phức tạp**
* AND-OR Tree Search - Cây tìm kiếm AND-OR
* Trong môi trường quan sát được một phần (Partially Observable Environments)
* Trong môi trường không xác định hoặc động (Unknown or Dynamic Environments - Tìm kiếm dựa trên niềm tin)

**Nhóm 5: Tìm kiếm thỏa mãn ràng buộc (Constraint Satisfaction Problems - CSPs)**
* Backtracking Search
* Forward Checking (trong CSP)
* AC-3 (Algorithm C-3 for arc consistency)

**Nhóm 6: Học tăng cường (Reinforcement Learning)**
* Q-Learning
* Temporal Difference (TD) Learning *(chưa được tích hợp sâu)*

---

## Minh Họa Các Thuật Toán Giải 8-Puzzle

Dưới đây là minh họa trực quan hoạt động của một số thuật toán nổi bật được triển khai để giải quyết bài toán 8-Puzzle:

### 1. A* (A Sao)
Thuật toán A* kết hợp chi phí đường đi đã qua ($g$) và một hàm heuristic ($h$) để ước tính chi phí đến đích, đảm bảo tìm được lời giải tối ưu nếu heuristic là chấp nhận được.

![Minh họa thuật toán A*](gif/Asao.gif)

### 2. Breadth-First Search (BFS) - Tìm kiếm theo chiều rộng
BFS duyệt qua các trạng thái theo từng tầng, đảm bảo tìm được lời giải ngắn nhất về số bước đi.

![Minh họa thuật toán BFS](gif/BFS.gif)

### 3. Beam Search
Beam Search là một biến thể của BFS, giới hạn số lượng trạng thái được xét ở mỗi độ sâu (beam width) để tiết kiệm bộ nhớ và thời gian, nhưng có thể không tìm ra lời giải tối ưu.

![Minh họa thuật toán Beam Search](gif/BeamSearch.gif)

### 4. Greedy Best-First Search - Tìm kiếm Tham lam
Thuật toán Greedy luôn chọn trạng thái kế tiếp có giá trị heuristic tốt nhất (gần đích nhất theo ước tính), giúp tìm lời giải nhanh nhưng không đảm bảo tối ưu.

![Minh họa thuật toán Greedy](gif/Greedy.gif)

### 5. Iterative Deepening A* (IDA*) - A* Sâu dần Lặp
IDA* kết hợp ưu điểm của A* (đánh giá bằng $f = g + h$) và tìm kiếm sâu dần (DFS), hiệu quả về bộ nhớ. Nó thực hiện một loạt các tìm kiếm DFS với ngưỡng chi phí $f$ tăng dần.

![Minh họa thuật toán IDA*](gif/IDA.gif)

### 6. Q-Learning (cho 8-Puzzle)
Q-Learning là một thuật toán học tăng cường không cần mô hình. Trong bối cảnh 8-Puzzle, agent học một chính sách (policy) để chọn hành động tối ưu từ mỗi trạng thái nhằm đạt được trạng thái đích.

![Minh họa Q-Learning giải 8-Puzzle](gif/Q-learning.gif)

### 7. Uniform Cost Search (UCS) - Tìm kiếm Chi phí Đồng nhất
UCS mở rộng trạng thái có chi phí đường đi ($g$) thấp nhất từ trạng thái bắt đầu, đảm bảo tìm được lời giải có tổng chi phí thấp nhất.

![Minh họa thuật toán UCS](gif/UCS.gif)

---

## Game Mê Cung: Chuột Tìm Phô Mai

Dự án cũng bao gồm một trò chơi mê cung nhỏ, nơi một chú chuột (agent) sử dụng thuật toán Q-Learning để học cách tìm và ăn phô mai một cách hiệu quả nhất. Bạn có thể huấn luyện agent và sau đó xem nó tự động chơi.

*(Nếu bạn có GIF cho trò chơi mê cung, bạn có thể thêm vào đây)*
---

## Cài Đặt và Chạy

**1. Yêu Cầu Hệ Thống:**
* Python 3.x
* CustomTkinter: `pip install customtkinter`
* Tkinter (thường được cài đặt sẵn với Python)
* Matplotlib: `pip install matplotlib`
* Pygame: `pip install pygame`
* NumPy: `pip install numpy`
* Pygame GUI (tùy chọn, nếu game mê cung sử dụng): `pip install pygame_gui`

**2. Các Tệp Cần Thiết:**
* `GiaoDien.py` (Tệp giao diện chính)
* `ThuatToan.py` (Tệp chứa các thuật toán giải đố)
* Thư mục `maze_game_module/` bao gồm:
    * `mouse_cheese_game.py`
    * `base_minigame.py`
    * `config.py`
    * Thư mục con `assets/` (bên trong `maze_game_module/`) chứa `mouse.png` và `cheese.png`.
* Thư mục `gif/` chứa các tệp GIF minh họa thuật toán.

**3. Cách Chạy Chương Trình:**

Để chạy giao diện chính của 8-Puzzle và truy cập game mê cung:
```bash
python GiaoDien.py
