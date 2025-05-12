# Project Giải Đố 8-Puzzle và Game Mê Cung Thông Minh

Chào mừng bạn đến với dự án kết hợp trình giải đố 8-Puzzle mạnh mẽ với nhiều thuật toán tìm kiếm khác nhau và một trò chơi mê cung (Chuột tìm Phô mai) được điều khiển bởi AI sử dụng Q-Learning!

Giao diện chính (`GiaoDien.py`) được xây dựng bằng CustomTkinter, cho phép bạn:
* Thiết lập trạng thái bắt đầu và trạng thái đích cho bài toán 8-Puzzle.
* Chọn từ một danh sách đa dạng các thuật toán tìm kiếm để giải đố.
* Theo dõi trực quan quá trình giải đố từng bước.
* Xem thống kê về hiệu suất của các thuật toán.
* Khởi chạy và xem agent Q-Learning chơi trò chơi Chuột tìm Phô mai.

## Các Thuật Toán gồm:

Nhóm 1 : Không có thông tin (Uninformed Search)
•	Breadth-First Search (BFS)  - tìm kiếm theo chiều rộng
•	Depth-First Search (DFS)  -  tìm kiếm theo chiều sâu
•	Uniform Cost Search (UCS)  - tìm kiếm theo chi phí thống nhất
•	Iterative Deepening DFS (IDS)  - tìm kiếm có giới hạn theo chiều sâu ( DFS + depth-limited Search )
Nhóm 2 : Có thông tin (Informed Search / Heuristic Search)
•	Greedy Search  ( tìm kiếm tham lam )
•	A*
•	Iterative Deepening A* (IDA*) tìm kiếm đào sâu có giới hạn ( DFS + depth-limited Search + A* ) 
Nhóm 3 : Local Search
•	Hill Climbing ( leo đồi )
o	Simple Hill Climbing   ( leo đồi đơn giản )
o	Steepest-Ascent Hill Climbing    ( leo đồi )
o	Stochastic Hill Climbing ( leo đồi ngẫu nhiên )
•	Simulated Annealing   ( ủ mô phỏng )
•	Genetic Algorithms  ( thuật toán di truyền )
•	Beam Search  ( tìm kiếm chùm tia )
Nhóm 4 : Tìm kiếm trong môi trường phức tạp
•	Tree Search AND – OR  ( Cây tìm kiếm And -Or
•	Partially Observable  ( nhìn thấy một phần )
•	Unknown or Dynamic Environment  ( Không nhìn thấy hoàn toàn – tìm kiếm trong môi trường niềm tin )
Nhóm 5 : Tìm kiếm trong môi trường có ràng buộc
•	Backtracking Search
•	Forward Checking
•	AC-3
Nhóm 6 : Reforcement Learning 
•	Q-Learning
•	Temporal Difference (TD) Learning ( cái này chưa rõ )

Dưới đây là minh họa hoạt động của một số thuật toán tìm kiếm được triển khai để giải quyết bài toán 8-Puzzle.
## Minh họa các thuật toán:

### 1. A* (A Sao)
Thuật toán A* kết hợp chi phí đường đi đã qua (g) và một hàm heuristic (h) để ước tính chi phí đến đích, đảm bảo tìm được lời giải tối ưu nếu heuristic là chấp nhận được.

![Minh họa thuật toán A*](gif/Asao.gif)

### 2. Breadth-First Search (BFS) - Tìm kiếm theo chiều rộng
BFS duyệt qua các trạng thái theo từng tầng, đảm bảo tìm được lời giải ngắn nhất về số bước đi.

![Minh họa thuật toán BFS](gif/BFS.gif)

### 3. Beam Search
Beam Search là một biến thể của BFS, giới hạn số lượng trạng thái được xét ở mỗi độ sâu (beam width) để tiết kiệm bộ nhớ và thời gian, nhưng có thể không tìm ra lời giải tối ưu.

![Minh họa thuật toán Beam Search](gif/BeamSearch.gif)

### 4. Greedy Best-First Search - Tìm kiếm Tham lam Tốt nhất đầu tiên
Thuật toán Greedy luôn chọn trạng thái kế tiếp có giá trị heuristic tốt nhất (gần đích nhất theo ước tính), giúp tìm lời giải nhanh nhưng không đảm bảo tối ưu.

![Minh họa thuật toán Greedy](gif/Greedy.gif)

### 5. Iterative Deepening A* (IDA*) - A* Sâu dần Lặp
IDA* kết hợp ưu điểm của A* (đánh giá bằng f = g + h) và tìm kiếm sâu dần (DFS), hiệu quả về bộ nhớ. Nó thực hiện một loạt các tìm kiếm DFS với ngưỡng chi phí f tăng dần.

![Minh họa thuật toán IDA*](gif/IDA.gif)

### 6. Q-Learning (cho 8-Puzzle)
Q-Learning là một thuật toán học tăng cường không cần mô hình. Trong bối cảnh 8-Puzzle, agent học một chính sách (policy) để chọn hành động tối ưu từ mỗi trạng thái nhằm đạt được trạng thái đích.

![Minh họa Q-Learning giải 8-Puzzle](gif/Q-learning.gif)

### 7. Uniform Cost Search (UCS) - Tìm kiếm Chi phí Đồng nhất
UCS mở rộng trạng thái có chi phí đường đi (g) thấp nhất từ trạng thái bắt đầu, đảm bảo tìm được lời giải có tổng chi phí thấp nhất.

![Minh họa thuật toán UCS](gif/UCS.gif)

## Game Mê Cung: Chuột Tìm Phô Mai

Dự án cũng bao gồm một trò chơi mê cung nhỏ, nơi một chú chuột (agent) sử dụng thuật toán Q-Learning để học cách tìm và ăn phô mai một cách hiệu quả nhất. Bạn có thể huấn luyện agent và sau đó xem nó tự động chơi.

*(Nếu bạn có GIF cho trò chơi mê cung, bạn có thể thêm vào đây)*
## Cài đặt và Chạy

1.  **Yêu cầu:**
    * Python 3.x
    * CustomTkinter (`pip install customtkinter`)
    * Tkinter (thường đi kèm với Python)
    * Matplotlib (`pip install matplotlib`)
    * Pygame (`pip install pygame`)
    * NumPy (`pip install numpy`)
    * Pygame GUI (`pip install pygame_gui`)

2.  **Các tệp cần thiết:**
    * `GiaoDien.py` (Giao diện chính)
    * `ThuatToan.py` (Chứa các thuật toán giải đố)
    * Thư mục `maze_game_module/` với các tệp:
        * `mouse_cheese_game.py`
        * `base_minigame.py` 
        * `config.py` 
        * Thư mục `assets/` bên trong `maze_game_module/` chứa `mouse.png` và `cheese.png`.
    * Thư mục `gif/` chứa các tệp GIF minh họa.

3.  **Để chạy giao diện chính của 8-Puzzle và game mê cung:**
    ```bash
    python GiaoDien.py
    ```
